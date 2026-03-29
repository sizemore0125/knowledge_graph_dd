import numpy as np
import torch


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
