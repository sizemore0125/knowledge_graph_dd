import numpy as np

DATA_DIR = "/home/logansizemore/Documents/knowledge_graph_dd/data/processed/"
# Relations: 63
# Edges: 35,649,195
# Entities: 5,864,272

import numpy as np
import torch


def load_data(path):
    with open(path, "r") as f:
        data = f.readlines()
    return [s.strip().replace("biolink:", "") for s in data]


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
    def __init__(self, num_edges, num_entities, num_datapoints: int = 65_000_000):
        self.num_edges = num_edges
        self.num_entites = num_entities

        self.num_datapoints = num_datapoints

    def __len__(self):
        return self.num_datapoints

    def __getitem__(self, idx):
        entity1 = torch.randint(0, self.num_entites, (1,))
        relation = torch.randint(0, self.num_edges, (1,))
        entity2 = torch.randint(0, self.num_entites, (1,))
        # y = torch.as_tensor(0).to(torch.float32)
        return torch.Tensor([entity1, relation, entity2]).to(torch.long)


class Model(torch.nn.Module):
    def __init__(self, n_entities, n_relations, emb_dim=8):
        super().__init__()
        self.entity_codebook = torch.nn.Embedding(n_entities, emb_dim)
        self.relation_codebook = torch.nn.Embedding(n_relations, emb_dim)
        self.emb_dim = emb_dim

        self.input_layer = torch.nn.Linear(emb_dim * 3, 512)
        self.output_layer = torch.nn.Linear(512, 1)

    def forward(self, edge):
        entity1 = self.entity_codebook(edge[:, 0])
        relation = self.relation_codebook(edge[:, 1])
        entity2 = self.entity_codebook(edge[:, 2])

        in_vector = torch.hstack([entity1, relation, entity2])

        z = self.input_layer(in_vector)
        a = torch.nn.functional.elu(z)
        y_pred = self.output_layer(a)

        assert y_pred.dtype == torch.float32

        return y_pred


def main():
    pos_dataset = PositiveDataset(DATA_DIR)
    neg_dataset = NegativeDataset(num_edges=63, num_entities=5_864_272, num_datapoints=len(pos_dataset))

    pos_dataloader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_size=32, shuffle=True)
    neg_dataloader = torch.utils.data.DataLoader(dataset=neg_dataset, batch_size=32, shuffle=True)

    model = Model(len(pos_dataset.entities_map), len(pos_dataset.relations_map))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for pos_edges, neg_edges in zip(pos_dataloader, neg_dataloader):
        pos_scores = model(pos_edges)
        neg_scores = model(neg_edges)

        loss = torch.nn.functional.softplus(-pos_scores).mean() + torch.nn.functional.softplus(neg_scores).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"LOSS: {loss.detach()}")


if __name__ == "__main__":
    main()
