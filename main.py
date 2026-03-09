import os

import igraph as ig
import numpy as np
import bmt

import torch
from tqdm import tqdm
from kgdd.rule_penalties import RulePenalties

DATA_DIR = "/home/logansizemore/Documents/knowledge_graph_dd/data/processed/"


def load_data(path):
    with open(path, "r") as f:
        data = f.readlines()
    return [s.strip().replace("biolink:", "") for s in data]


def load_hierarchy(path):
    edges = []

    with open(path, "r") as f:
        for line in f:
            child_str, parent_str = line.rstrip("\n").split("\t")
            child_id = int(child_str)
            parent_id = int(parent_str)
            edges.append((child_id, parent_id))

    return edges


def load_edges(path):
    return np.memmap(path, dtype=np.int32, mode="r").reshape(-1, 3)


def test_accuracy(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for edges, _, labels, _ in dataloader:
            edges = edges.to(device)
            labels = labels.to(device)

            logits = model(edges).squeeze(1)
            preds = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)

            correct += (preds == labels).sum().item()
            total += labels.numel()

    model.train()
    return correct / total if total > 0 else 0.0


class PositiveDataset(torch.utils.data.Dataset):
    def __init__(self, edges, relation_hierarchy: ig.Graph):
        self.edges = edges
        num_relations = relation_hierarchy.vcount()
        parent_ids = list(range(num_relations))
        for relation_id in range(num_relations):
            parents = relation_hierarchy.neighbors(relation_id, mode="OUT")
            if parents:
                parent_ids[relation_id] = min(parents)
        self.direct_relation_parents = np.asarray(parent_ids, dtype=np.int64)

    def __len__(self):
        return self.edges.shape[0]

    def __getitem__(self, idx):
        edge_np = np.asarray(self.edges[idx], dtype=np.int64)
        general_edge_np = edge_np.copy()
        general_edge_np[1] = self.direct_relation_parents[edge_np[1]]

        edge = torch.tensor(edge_np, dtype=torch.long)
        general_edge = torch.tensor(general_edge_np, dtype=torch.long)
        label = torch.tensor(1.0, dtype=torch.float32)
        task_id = torch.tensor(0, dtype=torch.long)

        return edge, general_edge, label, task_id


class NegativeDataset(torch.utils.data.Dataset):
    def __init__(self, edges):
        self.edges = edges

    def __len__(self):
        return self.edges.shape[0]

    def __getitem__(self, idx):
        edge_np = np.asarray(self.edges[idx], dtype=np.int64)
        edge = torch.tensor(edge_np, dtype=torch.long)

        dummy_edge = torch.tensor([-1, -1, -1], dtype=torch.long)
        label = torch.tensor(0.0, dtype=torch.float32)
        task_id = torch.tensor(0, dtype=torch.long)

        return edge, dummy_edge, label, task_id


class RelationSwapTestDataset(torch.utils.data.Dataset):
    def __init__(self, true_edges, relations_map):
        relation_to_id = {name: idx for idx, name in enumerate(relations_map)}
        treats_id = relation_to_id.get("treats")
        contra_id = relation_to_id.get("contraindicated_in")

        positive_edges = np.asarray(true_edges, dtype=np.int64)
        negative_edges = positive_edges.copy()
        is_treats = negative_edges[:, 1] == treats_id
        negative_edges[is_treats, 1] = contra_id
        negative_edges[~is_treats, 1] = treats_id

        all_edges = np.concatenate([positive_edges, negative_edges], axis=0)
        all_labels = np.concatenate(
            [
                np.ones(len(positive_edges), dtype=np.float32),
                np.zeros(len(negative_edges), dtype=np.float32),
            ]
        )

        self.edge_tensor = torch.tensor(all_edges, dtype=torch.long)
        self.label_tensor = torch.tensor(all_labels, dtype=torch.float32)

    def __len__(self):
        return self.edge_tensor.shape[0]

    def __getitem__(self, idx):
        return self.edge_tensor[idx], self.label_tensor[idx]


class DomainRangeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        edges: np.ndarray,
        entity_hierarchy: ig.Graph,
        relation_hierarchy: ig.Graph,
        num_samples: int,
    ) -> None:
        self.edges = edges
        self.entity_hierarchy = entity_hierarchy
        self.relation_hierarchy = relation_hierarchy
        self.toolkit = bmt.Toolkit()
        self.num_samples = num_samples

        self.relation_specs = self.build_relation_specs()
        if len(self.relation_specs) == 0:
            raise ValueError("DomainRangeDataset has no valid relation specs to sample from.")

    def normalize_name(self, name: str) -> str:
        return name.replace("biolink:", "")

    def expand_category(self, category_name: str) -> np.ndarray:
        node_ids = self.entity_hierarchy.subcomponent(category_name, mode="IN")
        return np.asarray(node_ids, dtype=np.int64)

    def build_relation_specs(self) -> list[dict]:
        specs = []
        all_entities = np.arange(self.entity_hierarchy.vcount(), dtype=np.int64)

        for relation_id, relation_name in tqdm(enumerate(self.relation_hierarchy.vs["name"]), desc="Characterizing DR Dataset:"):
            # Get Domain and Range names for Relation
            relation_slot = self.toolkit.get_element(relation_name)

            if relation_slot is None:
                continue

            if relation_slot.domain is None or relation_slot.range is None:
                continue

            domain_element = self.toolkit.get_element(relation_slot.domain)
            range_element = self.toolkit.get_element(relation_slot.range)

            if domain_element is None or range_element is None:
                continue

            domain_name = self.normalize_name(domain_element.class_uri)
            range_name = self.normalize_name(range_element.class_uri)

            # Get ID for Domain and Range
            try:
                domain_category_id = self.entity_hierarchy.vs.find(name=domain_name).index
                range_category_id = self.entity_hierarchy.vs.find(name=range_name).index
            except ValueError:
                continue

            # Get all entities that belong to domain/range categories.
            domain_pool = self.expand_category(domain_name)
            range_pool = self.expand_category(range_name)

            if len(domain_pool) == 0 or len(range_pool) == 0:
                continue

            # Get complement of domain/range
            out_domain_pool = all_entities[~np.isin(all_entities, domain_pool)]
            out_range_pool = all_entities[~np.isin(all_entities, range_pool)]

            if len(out_domain_pool) == 0 or len(out_range_pool) == 0:
                continue

            # Get all edges in KG with that relation.
            relation_mask = self.edges[:, 1] == relation_id
            graph_edges = np.asarray(self.edges[relation_mask], dtype=np.int64)

            if len(graph_edges) == 0:
                continue

            # Make knowledge graph edges a tuple.
            graph_edge_set = {(int(head_id), int(rel_id), int(tail_id)) for head_id, rel_id, tail_id in graph_edges}

            specs.append(
                {
                    "relation_id": relation_id,
                    "domain_pool": domain_pool,
                    "range_pool": range_pool,
                    "out_domain_pool": out_domain_pool,
                    "out_range_pool": out_range_pool,
                    "graph_edges": graph_edges,
                    "graph_edge_set": graph_edge_set,
                    "true_edge": np.asarray([domain_category_id, relation_id, range_category_id], dtype=np.int64),
                }
            )

        return specs

    def sample_in_graph(self, spec: dict) -> tuple[np.ndarray, float]:
        edge = spec["graph_edges"][np.random.randint(len(spec["graph_edges"]))]
        return edge, 1.0

    def sample_rule_valid_not_in_graph(self, spec: dict) -> tuple[np.ndarray, float]:
        relation_id = spec["relation_id"]
        for _ in range(100):
            head_id = int(spec["domain_pool"][np.random.randint(len(spec["domain_pool"]))])
            tail_id = int(spec["range_pool"][np.random.randint(len(spec["range_pool"]))])
            edge = (head_id, relation_id, tail_id)
            if edge not in spec["graph_edge_set"]:
                return np.asarray(edge, dtype=np.int64), 0.0
        return self.sample_rule_breaking_not_in_graph(spec)

    def sample_rule_breaking_not_in_graph(self, spec: dict) -> tuple[np.ndarray, float]:
        relation_id = spec["relation_id"]
        for _ in range(100):
            case_id = np.random.randint(3)
            if case_id == 0:
                head_id = int(spec["out_domain_pool"][np.random.randint(len(spec["out_domain_pool"]))])
                tail_id = int(spec["range_pool"][np.random.randint(len(spec["range_pool"]))])
            elif case_id == 1:
                head_id = int(spec["domain_pool"][np.random.randint(len(spec["domain_pool"]))])
                tail_id = int(spec["out_range_pool"][np.random.randint(len(spec["out_range_pool"]))])
            else:
                head_id = int(spec["out_domain_pool"][np.random.randint(len(spec["out_domain_pool"]))])
                tail_id = int(spec["out_range_pool"][np.random.randint(len(spec["out_range_pool"]))])

            edge = (head_id, relation_id, tail_id)
            if edge not in spec["graph_edge_set"]:
                return np.asarray(edge, dtype=np.int64), 0.0

        head_id = int(spec["out_domain_pool"][np.random.randint(len(spec["out_domain_pool"]))])
        tail_id = int(spec["out_range_pool"][np.random.randint(len(spec["out_range_pool"]))])
        return np.asarray([head_id, relation_id, tail_id], dtype=np.int64), 0.0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = self.relation_specs[np.random.randint(len(self.relation_specs))]
        draw = np.random.rand()

        if draw < 0.5:
            edge_np, label = self.sample_in_graph(spec)
        elif draw < 0.75:
            edge_np, label = self.sample_rule_valid_not_in_graph(spec)
        else:
            edge_np, label = self.sample_rule_breaking_not_in_graph(spec)

        edge = torch.tensor(edge_np, dtype=torch.long)
        true_edge = torch.tensor(spec["true_edge"], dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float32)
        task_id = torch.tensor(1, dtype=torch.long)
        return edge, true_edge, label, task_id


class HierarchicalEmbedding(torch.nn.Module):
    def __init__(self, num_nodes, dim, hierarchy):
        super().__init__()

        if not hierarchy.is_dag():
            raise ValueError("HierarchicalEmbedding requires a DAG (child -> parent).")

        self.residual = torch.nn.Embedding(num_nodes, dim)

        coeff_by_node = [dict() for _ in range(num_nodes)]
        topo_order = hierarchy.topological_sorting(mode="OUT")

        for node_id in reversed(topo_order):
            coeffs = {node_id: 1.0}
            parent_ids = hierarchy.neighbors(node_id, mode="OUT")

            if parent_ids:
                scale = 1.0 / len(parent_ids)
                for parent_id in parent_ids:
                    parent_coeffs = coeff_by_node[parent_id]
                    for ancestor_id, weight in parent_coeffs.items():
                        coeffs[ancestor_id] = coeffs.get(ancestor_id, 0.0) + scale * weight

            coeff_by_node[node_id] = coeffs

        flat_ancestor_ids = []
        flat_ancestor_weights = []
        offsets = [0]

        for node_id in range(num_nodes):
            coeffs = coeff_by_node[node_id]
            flat_ancestor_ids.extend(coeffs.keys())
            flat_ancestor_weights.extend(coeffs.values())
            offsets.append(len(flat_ancestor_ids))

        self.register_buffer(
            "ancestor_ids",
            torch.tensor(flat_ancestor_ids, dtype=torch.long),
        )

        self.register_buffer(
            "ancestor_weights",
            torch.tensor(flat_ancestor_weights, dtype=torch.float),
        )

        self.register_buffer(
            "offsets",
            torch.tensor(offsets, dtype=torch.long),
        )

    def forward(self, node_ids):
        flat_node_ids = node_ids.reshape(-1)

        unique_node_ids, inverse = torch.unique(
            flat_node_ids,
            sorted=False,
            return_inverse=True,
        )

        starts = self.offsets[unique_node_ids]
        ends = self.offsets[unique_node_ids + 1]
        counts = ends - starts

        owner_ids = torch.repeat_interleave(
            torch.arange(unique_node_ids.shape[0], device=node_ids.device),
            counts,
        )

        ancestor_ranges = [self.ancestor_ids[s:e] for s, e in zip(starts.tolist(), ends.tolist())]

        weight_ranges = [self.ancestor_weights[s:e] for s, e in zip(starts.tolist(), ends.tolist())]

        ancestor_index_tensor = torch.cat(ancestor_ranges)
        ancestor_weight_tensor = torch.cat(weight_ranges)

        ancestor_embeddings = self.residual(ancestor_index_tensor)
        weighted_ancestor_embeddings = ancestor_embeddings * ancestor_weight_tensor.unsqueeze(1)

        unique_embeddings = torch.zeros(
            (unique_node_ids.shape[0], self.residual.embedding_dim),
            device=node_ids.device,
            dtype=self.residual.weight.dtype,
        )

        unique_embeddings.index_add_(0, owner_ids, weighted_ancestor_embeddings)

        return unique_embeddings[inverse].reshape(
            *node_ids.shape,
            self.residual.embedding_dim,
        )


class Model(torch.nn.Module):
    def __init__(
        self,
        n_entities,
        n_relations,
        entity_hierarchy,
        relation_hierarchy,
        emb_dim=8,
    ):
        super().__init__()
        self.entity_codebook = HierarchicalEmbedding(n_entities, emb_dim, entity_hierarchy)
        self.relation_codebook = HierarchicalEmbedding(n_relations, emb_dim, relation_hierarchy)
        self.emb_dim = emb_dim

        self.input_layer = torch.nn.Linear(emb_dim * 3, 512)
        self.output_layer = torch.nn.Linear(512, 1)

    def forward(self, edge):

        entity_ids = torch.cat([edge[:, 0], edge[:, 2]], dim=0)
        entity_embeddings = self.entity_codebook(entity_ids)
        split_idx = edge.shape[0]

        entity1 = entity_embeddings[:split_idx]
        relation = self.relation_codebook(edge[:, 1])
        entity2 = entity_embeddings[split_idx:]

        in_vector = torch.hstack([entity1, relation, entity2])

        z = self.input_layer(in_vector)
        a = torch.nn.functional.elu(z)
        y_pred = self.output_layer(a)

        return y_pred


def main(num_epochs=3):
    entities_path = DATA_DIR + "entities.txt"
    relations_path = DATA_DIR + "relations.txt"

    entities_hierarchy_edge_list_path = DATA_DIR + "subclass_edge_list.txt"
    relations_hierarchy_edge_list_path = DATA_DIR + "relation_hierarchy_edge_list.txt"

    knowledge_graph_edges_path = DATA_DIR + "edges.bin"
    negative_edges_path = DATA_DIR + "negative_edges.bin"

    checkpoint_dir = "/home/logansizemore/Documents/knowledge_graph_dd/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(1, os.cpu_count() or 1)
    pin_memory = device.type == "cuda"

    entities_map = load_data(entities_path)
    relations_map = load_data(relations_path)
    num_entities = len(entities_map)
    num_relations = len(relations_map)

    entity_hierarchy_edges = load_hierarchy(entities_hierarchy_edge_list_path)
    relation_hierarchy_edges = load_hierarchy(relations_hierarchy_edge_list_path)

    entity_hierarchy = ig.Graph(n=num_entities, edges=entity_hierarchy_edges, directed=True)
    entity_hierarchy.vs["name"] = entities_map

    relation_hierarchy = ig.Graph(n=num_relations, edges=relation_hierarchy_edges, directed=True)
    relation_hierarchy.vs["name"] = relations_map

    positive_edges = load_edges(knowledge_graph_edges_path)
    negative_edges = load_edges(negative_edges_path)

    pos_dataset = PositiveDataset(
        edges=positive_edges,
        relation_hierarchy=relation_hierarchy,
    )
    neg_dataset = NegativeDataset(edges=negative_edges)

    train_pos_dataset, test_pos_dataset = torch.utils.data.random_split(pos_dataset, [len(pos_dataset) - 50000, 50000])
    train_neg_dataset, test_neg_dataset = torch.utils.data.random_split(neg_dataset, [len(neg_dataset) - 50000, 50000])

    train_pos_indices = np.asarray(train_pos_dataset.indices, dtype=np.int64)
    train_pos_edges = np.asarray(positive_edges[train_pos_indices], dtype=np.int32)

    domain_range_dataset = DomainRangeDataset(
        edges=train_pos_edges,
        entity_hierarchy=entity_hierarchy,
        relation_hierarchy=relation_hierarchy,
        num_samples=len(train_pos_dataset),
    )
    dataset = torch.utils.data.ConcatDataset([train_pos_dataset, train_neg_dataset, domain_range_dataset])
    test_dataset = torch.utils.data.ConcatDataset([test_pos_dataset, test_neg_dataset])

    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = Model(
        n_entities=num_entities,
        n_relations=num_relations,
        entity_hierarchy=entity_hierarchy,
        relation_hierarchy=relation_hierarchy,
    ).to(device)

    model.train()

    rule_penalties = RulePenalties(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch_idx in range(num_epochs):
        for step_idx, (edges, aux_edges, labels, task_ids) in enumerate(
            pbar := tqdm(
                train_dataloader,
                total=len(train_dataloader),
                miniters=50,
                mininterval=1,
                desc=f"Epoch {epoch_idx + 1}/{num_epochs}",
            )
        ):
            edges = edges.to(device, non_blocking=pin_memory)
            aux_edges = aux_edges.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory).unsqueeze(1)
            task_ids = task_ids.to(device, non_blocking=pin_memory)

            logits = model(edges)
            base_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

            kg_pos_mask = (task_ids == 0) & (labels.squeeze(1) > 0.5)
            if kg_pos_mask.any():
                pos_scores = logits[kg_pos_mask]
                general_edges = aux_edges[kg_pos_mask]
                general_scores = model(general_edges)

                # This enforces that P(e1 r1 e2) < P(e1 r2 e2), where r1 (specific relation) is a ancestor of r2 (general relation).
                hierarchy_penalty = torch.nn.functional.relu(pos_scores - general_scores).squeeze(1).mean()
            else:
                hierarchy_penalty = torch.tensor(0.0, device=device)

            domain_range_mask = task_ids == 1
            if domain_range_mask.any():
                rule_penalty = rule_penalties.domain_range_loss(
                    edges=edges[domain_range_mask],
                    true_edges=aux_edges[domain_range_mask],
                    logits=logits[domain_range_mask],
                )
            else:
                rule_penalty = torch.tensor(0.0, device=device)

            loss = base_loss + 0.25 * hierarchy_penalty + 0.25 * rule_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step_idx + 1) % 50 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    rel_hier=f"{hierarchy_penalty.item():.4f}",
                    rule=f"{rule_penalty.item():.4f}",
                    refresh=False,
                )

            if (step_idx) % 5000 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
                try:
                    acc = test_accuracy(model, test_dataloader, device)
                    print(f"test_accuracy={acc:.4f}")
                except Exception as exc:
                    print(f"test_accuracy_failed={exc}")
                torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
