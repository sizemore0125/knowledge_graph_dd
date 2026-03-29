import numpy as np
import torch


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
        dummy_edge = torch.tensor([-1, -1, -1], dtype=torch.long)
        task_id = torch.tensor(0, dtype=torch.long)
        return self.edge_tensor[idx], dummy_edge, self.label_tensor[idx], task_id
