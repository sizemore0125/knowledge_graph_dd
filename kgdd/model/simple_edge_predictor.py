import torch

from kgdd.model.hierarchical_embedding import DiskEmbeddings, HierarchicalEmbedding


class Model(torch.nn.Module):
    def __init__(
        self,
        n_entities,
        n_relations,
        entity_hierarchy,
        relation_hierarchy,
        emb_dim=32,
        random_inverse_edges: bool = False,
    ):
        super().__init__()
        self.entity_codebook = DiskEmbeddings(n_entities, emb_dim)
        self.relation_codebook = HierarchicalEmbedding(n_relations, emb_dim, relation_hierarchy)
        self.relation_direction = torch.nn.Embedding(2, emb_dim)
        self.emb_dim = emb_dim
        self.random_inverse_edges = random_inverse_edges

        self.input_layer = torch.nn.Linear(emb_dim * 3, 512)
        self.output_layer = torch.nn.Linear(512, 1)

    def forward(self, edge):
        entity1_ids = edge[:, 0]
        entity2_ids = edge[:, 2]
        rel_ids = edge[:, 1]

        if self.training and self.random_inverse_edges:
            swap_mask = torch.rand(edge.shape[0], device=edge.device) < 0.5
            swapped_entity1_ids = torch.where(swap_mask, entity2_ids, entity1_ids)
            swapped_entity2_ids = torch.where(swap_mask, entity1_ids, entity2_ids)
            entity1_ids = swapped_entity1_ids
            entity2_ids = swapped_entity2_ids

            direction_ids = swap_mask.to(torch.long)
        else:
            direction_ids = torch.zeros(edge.shape[0], device=edge.device, dtype=torch.long)

        relation = self.relation_codebook(rel_ids) + self.relation_direction(direction_ids)

        entity_ids = torch.cat([entity1_ids, entity2_ids], dim=0)
        entity_embeddings, _ = self.entity_codebook(entity_ids)
        split_idx = edge.shape[0]

        entity1 = entity_embeddings[:split_idx]
        entity2 = entity_embeddings[split_idx:]

        in_vector = torch.hstack([entity1, relation, entity2])

        z = self.input_layer(in_vector)
        a = torch.nn.functional.elu(z)
        y_pred = self.output_layer(a)

        return y_pred
