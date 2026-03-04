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


def load_hierarchy(path, num_nodes):
    edges = []

    with open(path, "r") as f:
        for line in f:
            child_str, parent_str = line.rstrip("\n").split("\t")
            child_id = int(child_str)
            parent_id = int(parent_str)

            if child_id < num_nodes and parent_id < num_nodes:
                edges.append((child_id, parent_id))

    return edges


def build_direct_relation_parents(num_relations, relation_hierarchy_edges):
    graph = ig.Graph(n=num_relations, edges=relation_hierarchy_edges, directed=True)
    parent_ids = list(range(num_relations))

    for relation_id in range(num_relations):
        parents = graph.neighbors(relation_id, mode="OUT")
        if parents:
            parent_ids[relation_id] = min(parents)

    return torch.tensor(parent_ids, dtype=torch.long)


class PositiveDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        edges_path = data_dir + "edges.bin"

        self.edges = np.memmap(edges_path, dtype=np.int32, mode="r").reshape(-1, 3)

        self.entities_map = load_data(data_dir + "entities.txt")
        self.relations_map = load_data(data_dir + "relations.txt")

    def __len__(self):
        return self.edges.shape[0]

    def __getitem__(self, idx):
        edge = torch.Tensor(np.array(self.edges[idx])).to(torch.long)
        # y = torch.as_tensor(1).to(torch.float32)
        return edge


class NegativeDataset(torch.utils.data.Dataset):
    def __init__(self, num_relations, num_entities, num_datapoints):
        self.num_relations = num_relations
        self.num_entities = num_entities

        self.num_datapoints = num_datapoints

    def __len__(self):
        return self.num_datapoints

    def __getitem__(self, idx):
        entity1 = torch.randint(0, self.num_entities, (1,))
        relation = torch.randint(0, self.num_relations, (1,))
        entity2 = torch.randint(0, self.num_entities, (1,))
        # y = torch.as_tensor(0).to(torch.float32)
        return torch.Tensor([entity1, relation, entity2]).to(torch.long)


class HierarchicalEmbedding(torch.nn.Module):
    def __init__(self, num_nodes, dim, edges):
        super().__init__()
        self.num_nodes = num_nodes
        self.residual = torch.nn.Embedding(num_nodes, dim)
        graph = ig.Graph(n=num_nodes, edges=edges, directed=True)

        if not graph.is_dag():
            raise ValueError("HierarchicalEmbedding requires a DAG (child -> parent).")

        # Precompute coefficients for:
        # E(v) = Delta(v) + mean(E(parent(v)))
        coeff_by_node = [dict() for _ in range(num_nodes)]
        topo_order = graph.topological_sorting(mode="OUT")

        for node_id in reversed(topo_order):
            coeffs = {node_id: 1.0}
            parent_ids = graph.neighbors(node_id, mode="OUT")
            if parent_ids:
                scale = 1.0 / len(parent_ids)
                for parent_id in parent_ids:
                    parent_coeffs = coeff_by_node[parent_id]
                    for ancestor_id, weight in parent_coeffs.items():
                        coeffs[ancestor_id] = coeffs.get(ancestor_id, 0.0) + scale * weight
            coeff_by_node[node_id] = coeffs

        self.ancestor_ids_by_node = [list(coeffs.keys()) for coeffs in coeff_by_node]
        self.ancestor_weights_by_node = [list(coeffs.values()) for coeffs in coeff_by_node]

    def forward(self, node_ids):
        flat_node_ids = node_ids.reshape(-1)
        unique_node_ids, inverse = torch.unique(flat_node_ids, sorted=False, return_inverse=True)
        unique_node_list = unique_node_ids.detach().cpu().tolist()

        flat_ancestor_ids = []
        flat_ancestor_weights = []
        owner_ids = []

        for batch_idx, node_id in enumerate(unique_node_list):
            ancestor_ids = self.ancestor_ids_by_node[node_id]
            ancestor_weights = self.ancestor_weights_by_node[node_id]
            flat_ancestor_ids.extend(ancestor_ids)
            flat_ancestor_weights.extend(ancestor_weights)
            owner_ids.extend([batch_idx] * len(ancestor_ids))

        ancestor_index_tensor = torch.tensor(
            flat_ancestor_ids,
            device=flat_node_ids.device,
            dtype=torch.long,
        )
        ancestor_weight_tensor = torch.tensor(
            flat_ancestor_weights,
            device=flat_node_ids.device,
            dtype=self.residual.weight.dtype,
        )
        owner_index_tensor = torch.tensor(
            owner_ids,
            device=flat_node_ids.device,
            dtype=torch.long,
        )

        ancestor_embeddings = self.residual(ancestor_index_tensor)
        weighted_ancestor_embeddings = ancestor_embeddings * ancestor_weight_tensor.unsqueeze(1)

        unique_embeddings = torch.zeros(
            (unique_node_ids.shape[0], self.residual.embedding_dim),
            device=flat_node_ids.device,
            dtype=self.residual.weight.dtype,
        )
        unique_embeddings.index_add_(0, owner_index_tensor, weighted_ancestor_embeddings)

        return unique_embeddings[inverse].reshape(*node_ids.shape, self.residual.embedding_dim)


class Model(torch.nn.Module):
    def __init__(
        self,
        n_entities,
        n_relations,
        entity_hierarchy_edges,
        relation_hierarchy_edges,
        emb_dim=8,
    ):
        super().__init__()
        self.entity_codebook = HierarchicalEmbedding(n_entities, emb_dim, entity_hierarchy_edges)
        self.relation_codebook = HierarchicalEmbedding(n_relations, emb_dim, relation_hierarchy_edges)
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
    pos_dataset = PositiveDataset(DATA_DIR)
    neg_dataset = NegativeDataset(
        num_relations=len(pos_dataset.relations_map),
        num_entities=len(pos_dataset.entities_map),
        num_datapoints=len(pos_dataset),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(6, os.cpu_count() or 1)
    pin_memory = device.type == "cuda"

    pos_dataloader = torch.utils.data.DataLoader(
        dataset=pos_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    neg_dataloader = torch.utils.data.DataLoader(
        dataset=neg_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    entity_hierarchy_edges = load_hierarchy(
        DATA_DIR + "subclass_edge_list.txt",
        len(pos_dataset.entities_map),
    )
    relation_hierarchy_edges = load_hierarchy(
        DATA_DIR + "relation_hierarchy_edge_list.txt",
        len(pos_dataset.relations_map),
    )
    direct_relation_parents = build_direct_relation_parents(
        len(pos_dataset.relations_map),
        relation_hierarchy_edges,
    ).to(device)

    model = Model(
        n_entities=len(pos_dataset.entities_map),
        n_relations=len(pos_dataset.relations_map),
        entity_hierarchy_edges=entity_hierarchy_edges,
        relation_hierarchy_edges=relation_hierarchy_edges,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for step_idx, (pos_edges, neg_edges) in enumerate(
        pbar := tqdm(
            zip(pos_dataloader, neg_dataloader),
            total=len(pos_dataloader),
            miniters=10,
            mininterval=0.5,
        )
    ):
        pos_edges = pos_edges.to(device, non_blocking=pin_memory)
        neg_edges = neg_edges.to(device, non_blocking=pin_memory)
        pos_scores = model(pos_edges)
        neg_scores = model(neg_edges)

        general_edges = pos_edges.clone()
        general_relation_ids = direct_relation_parents[pos_edges[:, 1]]
        general_edges[:, 1] = general_relation_ids
        general_scores = model(general_edges)

        base_loss = torch.nn.functional.softplus(-pos_scores).mean() + torch.nn.functional.softplus(neg_scores).mean()

        # This enforces that P(e1 r1 e2) < P(e1 r2 e2), where r1 (specific relation) is a ancestor of r2 (general relation).
        hierarchy_penalty = torch.nn.functional.relu(pos_scores - general_scores).squeeze(1).mean()

        loss = base_loss + 1.0 * hierarchy_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step_idx + 1) % 10 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", rel_hier=f"{hierarchy_penalty.item():.4f}", refresh=False)


if __name__ == "__main__":
    main()
