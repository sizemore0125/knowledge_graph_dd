import torch

from kgdd.model.hierarchical_embedding import DiskEmbeddings, HierarchicalEmbedding


class TripleNeighborhoodAttention(torch.nn.Module):
    def __init__(
        self,
        entity_codebook: DiskEmbeddings,
        relation_codebook: HierarchicalEmbedding,
        relation_direction: torch.nn.Embedding,
        hidden_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.entity_codebook = entity_codebook
        self.relation_codebook = relation_codebook
        self.relation_direction = relation_direction

        entity_dim = entity_codebook.centers.embedding_dim
        relation_dim = relation_codebook.residual.embedding_dim
        direction_dim = relation_direction.embedding_dim

        self.ctx_proj = torch.nn.Linear(entity_dim + relation_dim + direction_dim, hidden_dim)
        self.query_proj = torch.nn.Linear(entity_dim, hidden_dim)

        self.attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = torch.nn.LayerNorm(hidden_dim)

    def build_context_tokens(
        self,
        neighbor_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        direction_ids: torch.Tensor,
    ):
        neighbors, _ = self.entity_codebook(neighbor_ids)
        rel = self.relation_codebook(relation_ids)
        direction = self.relation_direction(direction_ids)

        ctx = torch.cat([neighbors, rel, direction], dim=-1)
        return self.ctx_proj(ctx)

    def encode_endpoint(
        self,
        entity_ids: torch.Tensor,
        neighbor_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        direction_ids: torch.Tensor,
    ):
        ent, _ = self.entity_codebook(entity_ids)
        q = self.query_proj(ent).unsqueeze(1)

        seq = self.build_context_tokens(
            neighbor_ids=neighbor_ids,
            relation_ids=relation_ids,
            direction_ids=direction_ids,
        )

        ctx, _ = self.attn(q, seq, seq, need_weights=False)
        return self.norm(q.squeeze(1) + ctx.squeeze(1))

    def forward(
        self,
        e1_ids: torch.Tensor,
        e1_neighbor_ids: torch.Tensor,
        e1_relation_ids: torch.Tensor,
        e1_direction_ids: torch.Tensor,
        e2_ids: torch.Tensor,
        e2_neighbor_ids: torch.Tensor,
        e2_relation_ids: torch.Tensor,
        e2_direction_ids: torch.Tensor,
    ):
        e1_out = self.encode_endpoint(
            entity_ids=e1_ids,
            neighbor_ids=e1_neighbor_ids,
            relation_ids=e1_relation_ids,
            direction_ids=e1_direction_ids,
        )

        e2_out = self.encode_endpoint(
            entity_ids=e2_ids,
            neighbor_ids=e2_neighbor_ids,
            relation_ids=e2_relation_ids,
            direction_ids=e2_direction_ids,
        )

        return e1_out, e2_out


class Model(torch.nn.Module):
    def __init__(
        self,
        n_entities,
        n_relations,
        entity_hierarchy,
        relation_hierarchy,
        emb_dim=32,
        random_inverse_edges: bool = True,
        use_attention: bool = True,
        attention_hidden_dim: int = 32,
        attention_num_heads: int = 4,
    ):
        super().__init__()
        self.entity_codebook = DiskEmbeddings(n_entities, emb_dim)
        self.relation_codebook = HierarchicalEmbedding(n_relations, emb_dim, relation_hierarchy)
        self.relation_direction = torch.nn.Embedding(2, emb_dim)
        self.emb_dim = emb_dim
        self.random_inverse_edges = random_inverse_edges
        self.use_attention = use_attention

        self.attention_hidden_dim = attention_hidden_dim
        self.attention_num_heads = attention_num_heads

        if self.use_attention:
            self.attention = TripleNeighborhoodAttention(
                entity_codebook=self.entity_codebook,
                relation_codebook=self.relation_codebook,
                relation_direction=self.relation_direction,
                hidden_dim=self.attention_hidden_dim,
                num_heads=self.attention_num_heads,
            )
            input_dim = self.attention_hidden_dim * 2 + emb_dim
        else:
            input_dim = emb_dim * 3

        self.input_layer = torch.nn.Linear(input_dim, 512)
        self.output_layer = torch.nn.Linear(512, 1)

    def forward(
        self,
        edge,
        e1_neighbor_ids=None,
        e1_relation_ids=None,
        e1_direction_ids=None,
        e2_neighbor_ids=None,
        e2_relation_ids=None,
        e2_direction_ids=None,
    ):
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

        if self.use_attention:
            entity1, entity2 = self.attention(
                e1_ids=entity1_ids,
                e1_neighbor_ids=e1_neighbor_ids,
                e1_relation_ids=e1_relation_ids,
                e1_direction_ids=e1_direction_ids,
                e2_ids=entity2_ids,
                e2_neighbor_ids=e2_neighbor_ids,
                e2_relation_ids=e2_relation_ids,
                e2_direction_ids=e2_direction_ids,
            )
        else:
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
