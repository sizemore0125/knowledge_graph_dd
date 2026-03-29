import numpy as np
import torch


class PositiveDataset(torch.utils.data.Dataset):
    def __init__(self, edges, relation_hierarchy):
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
