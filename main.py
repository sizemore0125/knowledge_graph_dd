import os

import igraph as ig
import numpy as np

import torch
from tqdm import tqdm

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


def build_direct_relation_parents(num_relations, relation_hierarchy):
    parent_ids = list(range(num_relations))

    for relation_id in range(num_relations):
        parents = relation_hierarchy.neighbors(relation_id, mode="OUT")
        if parents:
            parent_ids[relation_id] = min(parents)

    return torch.tensor(parent_ids, dtype=torch.long)


class PositiveDataset(torch.utils.data.Dataset):
    def __init__(self, edges):
        self.edges = edges

    def __len__(self):
        return self.edges.shape[0]

    def __getitem__(self, idx):
        edge = torch.tensor(self.edges[idx], dtype=torch.long)
        label = torch.tensor(1.0, dtype=torch.float32)
        return edge, label


class NegativeDataset(torch.utils.data.Dataset):
    def __init__(self, num_relations, num_entities, num_datapoints):
        self.num_relations = num_relations
        self.num_entities = num_entities

        self.num_datapoints = num_datapoints

    def __len__(self):
        return self.num_datapoints

    def __getitem__(self, idx):
        entity1 = torch.randint(0, self.num_entities, ()).item()
        relation = torch.randint(0, self.num_relations, ()).item()
        entity2 = torch.randint(0, self.num_entities, ()).item()
        edge = torch.tensor([entity1, relation, entity2], dtype=torch.long)
        label = torch.tensor(0.0, dtype=torch.float32)
        return edge, label


class DomainRangeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        edges: np.ndarray,
        entity_hierarchy: ig.Graph,
        relation_hierarchy: ig.Graph,
        relation_id: str,
        domain_labels: list[str],
        range_labels: list[str],
        num_samples: int,
        mode: str,
    ) -> None:
        self.edges = edges
        self.relation_id = relation_id
        self.num_samples = num_samples
        self.mode = mode

        relation_ids = np.asarray(
            relation_hierarchy.subcomponent(relation_id, mode="IN"),
            dtype=np.int32,
        )
        relation_mask = np.isin(self.edges[:, 1], relation_ids)
        self.pos_graph_edges = self.edges[relation_mask]

        self.domain_pool = self.expand_labels(entity_hierarchy, domain_labels)
        self.range_pool = self.expand_labels(entity_hierarchy, range_labels)

        all_entities = np.arange(entity_hierarchy.vcount(), dtype=np.int64)
        self.out_domain_pool = all_entities[~np.isin(all_entities, self.domain_pool)]
        self.out_range_pool = all_entities[~np.isin(all_entities, self.range_pool)]

    def expand_labels(self, hierarchy: ig.Graph, labels: list[str]) -> np.ndarray:
        node_ids = []
        for label in labels:
            root_id = hierarchy.vs.find(name=label).index
            node_ids.extend(hierarchy.subcomponent(root_id, mode="IN"))
        return np.unique(np.asarray(node_ids, dtype=np.int64))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "positive":
            if np.random.rand() < 0.5 and len(self.pos_graph_edges) > 0:
                edge = self.pos_graph_edges[np.random.randint(len(self.pos_graph_edges))]
                return torch.tensor(edge, dtype=torch.long), torch.tensor(1.0, dtype=torch.float32)
            else:
                head_id = self.domain_pool[np.random.randint(len(self.domain_pool))]
                tail_id = self.range_pool[np.random.randint(len(self.range_pool))]
                edge = np.array([head_id, self.relation_id, tail_id], dtype=np.int64)
                return torch.tensor(edge, dtype=torch.long), torch.tensor(0.0, dtype=torch.float32)

        case_id = np.random.randint(3)
        if case_id == 0:
            head_id = self.out_domain_pool[np.random.randint(len(self.out_domain_pool))]
            tail_id = self.range_pool[np.random.randint(len(self.range_pool))]
        elif case_id == 1:
            head_id = self.domain_pool[np.random.randint(len(self.domain_pool))]
            tail_id = self.out_range_pool[np.random.randint(len(self.out_range_pool))]
        else:
            head_id = self.out_domain_pool[np.random.randint(len(self.out_domain_pool))]
            tail_id = self.out_range_pool[np.random.randint(len(self.out_range_pool))]

        edge = torch.tensor([head_id, self.relation_id, tail_id], dtype=torch.long)
        return edge, torch.tensor(0.0, dtype=torch.float32)


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


def main():
    entities_path = DATA_DIR + "entities.txt"
    relations_path = DATA_DIR + "relations.txt"

    entities_hierarchy_edge_list_path = DATA_DIR + "subclass_edge_list.txt"
    relations_hierarchy_edge_list_path = DATA_DIR + "relation_hierarchy_edge_list.txt"

    knowledge_graph_edges_path = DATA_DIR + "edges.bin"

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

    knowledge_graph_edges = load_edges(knowledge_graph_edges_path)

    pos_dataset = PositiveDataset(edges=knowledge_graph_edges)
    neg_dataset = NegativeDataset(
        num_relations=num_relations,
        num_entities=num_entities,
        num_datapoints=len(pos_dataset),
    )
    dataset = torch.utils.data.ConcatDataset([pos_dataset, neg_dataset])

    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=512,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    direct_relation_parents = build_direct_relation_parents(
        num_relations,
        relation_hierarchy,
    ).to(device)

    model = Model(
        n_entities=num_entities,
        n_relations=num_relations,
        entity_hierarchy=entity_hierarchy,
        relation_hierarchy=relation_hierarchy,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for step_idx, (edges, labels) in enumerate(
        pbar := tqdm(
            train_dataloader,
            total=len(train_dataloader),
            miniters=50,
            mininterval=1,
        )
    ):
        edges = edges.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory).unsqueeze(1)

        logits = model(edges)
        base_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

        pos_mask = labels.squeeze(1) > 0.5
        if pos_mask.any():
            pos_edges = edges[pos_mask]
            pos_scores = logits[pos_mask]

            general_edges = pos_edges.clone()
            general_relation_ids = direct_relation_parents[pos_edges[:, 1]]
            general_edges[:, 1] = general_relation_ids
            general_scores = model(general_edges)

            # This enforces that P(e1 r1 e2) < P(e1 r2 e2), where r1 (specific relation) is a ancestor of r2 (general relation).
            hierarchy_penalty = torch.nn.functional.relu(pos_scores - general_scores).squeeze(1).mean()
        else:
            hierarchy_penalty = torch.tensor(0.0, device=device)

        loss = base_loss + 1.0 * hierarchy_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step_idx + 1) % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", rel_hier=f"{hierarchy_penalty.item():.4f}", refresh=False)


if __name__ == "__main__":
    main()
