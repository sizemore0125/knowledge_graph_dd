import torch

from kgdd.model.hierarchical_embedding import HierarchicalEmbedding


class Model(torch.nn.Module):
    def __init__(
        self,
        n_entities,
        n_relations,
        entity_hierarchy,
        relation_hierarchy,
        emb_dim=32,
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
