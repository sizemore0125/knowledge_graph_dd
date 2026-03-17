"""
Recurrent Message Passing Graph Neural Network trained for relation detection between drug and disease (specifically relation 'cures')
Using an initial embedding of the drug and disease in a 'vertex space' which is enhanced by knowledge of existing paths 
between the drug and disease of interest, which is then predicted whether a drug and disease should have an edge with 'cures'
between them. 

Written for final project for Stephen Ramsey's Biological Networks class
by Cameron Hatler and Logan Sizemore
"""
import os

import igraph as ig
import numpy as np
import bmt

import torch
import torch.nn as nn
from tqdm import tqdm
from kgdd.rule_penalties import RulePenalties


DATA_DIR = "./data/processed/"
CHECKPOINT_DIR = "./checkpoints"
# Relations: 63
# Edges: 35,649,195
# Entities: 5,864,272

def load_data(path):
    with open(path, "r", encoding="UTF-8") as f:
        data = f.readlines()
    return data

def collate_fn(batch):
    """
    Re-pads variable-length node/relation sequences within a batch
    using -1 as the padding value.
    """
    nodes_list, relations_list = zip(*batch)

    max_node_len = max(n.size(0) for n in nodes_list)
    max_rel_len  = max(r.size(0) for r in relations_list)

    padded_nodes     = torch.zeros(len(nodes_list),     max_node_len, dtype=torch.long)
    padded_relations = torch.zeros(len(relations_list), max_rel_len,  dtype=torch.long)
    lengths = torch.zeros(len(nodes_list), dtype=torch.long)

    for i, (n, r) in enumerate(zip(nodes_list, relations_list)):
        padded_nodes[i,     :n.size(0)] = n
        padded_relations[i, :r.size(0)] = r
        lengths[i] = r.size(0)

    return padded_nodes, padded_relations, lengths

def load_hierarchy(path):
    edges = []

    with open(path, "r") as f:
        for line in f:
            child_str, parent_str = line.rstrip("\n").split("\t")
            child_id = int(child_str)
            parent_id = int(parent_str)
            edges.append((child_id, parent_id))

    return edges


def test_accuracy(model, pos_dataloader, neg_dataloader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for vertices, relations, lengths in pos_dataloader:
            vertices = vertices.to(device)
            relations = relations.to(device)
            lengths = lengths.to(device)

            scores = model(vertices, relations, lengths).squeeze(1)
            preds = (scores >= 0.5).int()

            correct += (preds == 1).sum().item()
            total += preds.numel() 

        for vertices, relations, lengths in neg_dataloader:
            vertices = vertices.to(device)
            relations = relations.to(device)
            lengths = lengths.to(device)

            scores = model(vertices, relations, lengths).squeeze(1)
            preds = (scores >= 0.5).int()

            correct += (preds == 0).sum().item()
            total += preds.numel()    

    model.train()
    return correct / total if total > 0 else 0.0


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label, path_len):

        self.nodes = np.memmap(data_dir + label + "_path_nodes.bin", dtype=np.int32, mode="r").reshape(-1, path_len+1)
        self.relations = np.memmap(data_dir + label + "_path_relations.bin", dtype=np.int32, mode="r").reshape(-1, path_len)

        self.entities_map = load_data(data_dir + "entities.txt")
        self.relations_map = load_data(data_dir + "relations.txt")        

    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, idx):
        nodes = torch.from_numpy(self.nodes[idx].copy()).long()
        relations = torch.from_numpy(self.relations[idx].copy()).long()

        # Find the first occurrence of -1 in the nodes tensor
        non_padding_length = (nodes != -1).sum().item()

        # Slice the tensors to remove the padding
        nodes = nodes[:non_padding_length]
        relations = relations[:non_padding_length-1]


        return nodes, relations

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

class ResidualPathStep(nn.Module):
    """
    Model that learns residual weights along a path
    from a source to a target by gathering information
    of nodes and relations along path
    """
    
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, node_i, rel_i, node_j):
        """
        node_i: [batch, emb_dim]
        rel_i:  [batch, emb_dim]
        node_j: [batch, emb_dim]
        """
        
        x = torch.cat([node_i, rel_i, node_j], dim=-1)
        residual = self.net(x)
        
        return residual

class PathReasoningModel(nn.Module):
    """
    Class used to find residuals along an entire path
    between two vertices, work is done by ResidualPathStep
    model for each step along the path
    """
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        
        self.step = ResidualPathStep(emb_dim, hidden_dim)
        
    def forward(self, node_embeddings, relation_embeddings, lengths):
        """
        node_embeddings: [batch, path_len+1, emb_dim]
        relation_embeddings: [batch, path_len, emb_dim]
        node_mask: [batch, path_len+1]
        rel_mask: [batch, path_len]
        """
        batch_size, _, emb_dim = node_embeddings.shape
        next_node = node_embeddings[:,0]  # starting node
        residual = torch.zeros_like(next_node)

        for i in range(relation_embeddings.shape[1]):
            active = (i < lengths)
            
            h = next_node + residual
            r = relation_embeddings[:, i]
            next_node = node_embeddings[:, i+1]
            new_residual = self.step(h, r, next_node)

            # Only update residual for samples that have a real step here
            residual = torch.where(active.unsqueeze(-1), new_residual, residual)
            
        return residual
    

class EdgePredictor(nn.Module):
    """
    The model that takes the final embedding
    to determine if there should be an edge
    between the two nodes
    """
    def __init__(self, emb_dim):
        super().__init__()
        
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, 1)
        )
    
    def forward(self, h_path):
        score = self.scorer(h_path)
        return torch.sigmoid(score)
    

class KnowledgeGraphPathModel(nn.Module):
    """
    Putting the path reasoning and Edge Prediction together
    to predict if two nodes should be related based on a 
    different existing path between them
    """
    def __init__(self, n_entities, n_relations, entity_hierarchy, relation_hierarchy, hidden_dim=512, emb_dim=32):
        super().__init__()
        
        self.entity_codebook = HierarchicalEmbedding(n_entities + 2, emb_dim, entity_hierarchy)
        self.relation_codebook = HierarchicalEmbedding(n_relations, emb_dim, relation_hierarchy)

        self.path_model = PathReasoningModel(emb_dim, hidden_dim)
        self.edge_predictor = EdgePredictor(emb_dim)
        self.n_entities = n_entities
        
    def forward(self, nodes, relations, lengths):
        """
        :param nodes: list of node embeddings along path
        :param relations: list of relations along path
        """
        emb_nodes = self.entity_codebook(nodes)
        emb_relat = self.relation_codebook(relations)
        path_embedding = self.path_model(emb_nodes, emb_relat, lengths)

        score = self.edge_predictor(path_embedding)
    
        return score
    
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    entities_path = DATA_DIR + "entities.txt"
    relations_path = DATA_DIR + "relations.txt"

    entities_hierarchy_edge_list_path = DATA_DIR + "subclass_edge_list.txt"
    relations_hierarchy_edge_list_path = DATA_DIR + "relation_hierarchy_edge_list.txt"

    knowledge_graph_edges_path = DATA_DIR + "edges.bin"
    negative_edges_path = DATA_DIR + "negative_edges.bin"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(6, os.cpu_count() or 1)
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

    pos_dataset = Dataset(DATA_DIR, 'pos', 3)
    neg_dataset = Dataset(DATA_DIR, 'neg', 3)

    model = KnowledgeGraphPathModel(
        n_entities=num_entities,
        n_relations=num_relations,
        entity_hierarchy=entity_hierarchy,
        relation_hierarchy=relation_hierarchy
    ).to(device)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_pos_dataset, test_pos_dataset = torch.utils.data.random_split(pos_dataset, [len(pos_dataset) - 500_000, 500_000])
    train_neg_dataset, test_neg_dataset = torch.utils.data.random_split(neg_dataset, [len(neg_dataset) - 500_000, 500_000])

    train_pos_dataloader = torch.utils.data.DataLoader(
        dataset=train_pos_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
    train_neg_dataloader = torch.utils.data.DataLoader(
        dataset=train_neg_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    test_pos_dataloader = torch.utils.data.DataLoader(
        dataset=test_pos_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    test_neg_dataloader = torch.utils.data.DataLoader(
        dataset=test_neg_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    rule_penalties = RulePenalties(model)
    num_epochs = 25
    last_eval_prefix = ""

    for epoch_idx in range(num_epochs):
        epoch_desc = f"{last_eval_prefix}Epoch {epoch_idx + 1}/{num_epochs}".strip()

        for step_idx, (pos_paths, neg_paths) in enumerate(pbar := tqdm(  
                                                            zip(train_pos_dataloader, train_neg_dataloader),
                                                            total=min(len(train_pos_dataloader),len(train_neg_dataloader)), 
                                                            miniters=50,mininterval=1,desc=epoch_desc)):

            pos_vertices, pos_relations, pos_lengths = pos_paths
            neg_vertices, neg_relations, neg_lengths = neg_paths

            # anonymize first and last vertex along the path
            pos_vertices[:,0] = len(pos_dataset.entities_map)
            pos_vertices[:,-1] = len(pos_dataset.entities_map) + 1

            # anonymize first and last vertex along path for negative set
            neg_vertices[:,0] = len(neg_dataset.entities_map)
            neg_vertices[:,-1] = len(neg_dataset.entities_map) + 1

            # send to device
            pos_vertices = pos_vertices.to(device, non_blocking=pin_memory)
            pos_relations = pos_relations.to(device,non_blocking=pin_memory)
            pos_lengths = pos_lengths.to(device, non_blocking=pin_memory)
            neg_vertices = neg_vertices.to(device, non_blocking=pin_memory)
            neg_relations = neg_relations.to(device, non_blocking=pin_memory)
            neg_lengths = neg_lengths.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()

            pos_scores = model(pos_vertices, pos_relations, pos_lengths)
            neg_scores = model(neg_vertices, neg_relations, neg_lengths)
            base_loss = torch.nn.functional.softplus(-pos_scores).mean() + torch.nn.functional.softplus(neg_scores).mean()

            loss = base_loss
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if (step_idx + 1) % 50 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    refresh=False,
                )

            if (step_idx) % 5000 == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "model.pt")
                acc = test_accuracy(model, test_pos_dataloader, test_neg_dataloader, device)
                last_eval_prefix = f"[test={acc:.4f}]"
                pbar.set_description(f"{last_eval_prefix}Epoch {epoch_idx + 1}/{num_epochs}")
                torch.save(model.state_dict(), checkpoint_path)




if __name__ == "__main__":
    main()