import torch


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


class DiskEmbeddings(torch.nn.Module):
    def __init__(self, num_nodes: int, dim: int):
        super().__init__()

        self.num_nodes = num_nodes
        self.dim = dim

        self.centers = torch.nn.Embedding(num_nodes, dim)
        self.radii = torch.nn.Embedding(num_nodes, 1)

    def forward(self, node_ids: torch.LongTensor):
        centers = self.centers(node_ids)
        radii = self.radii(node_ids).squeeze(-1)
        return centers, radii

    def score(
        self,
        parent_ids: torch.LongTensor,
        child_ids: torch.LongTensor,
    ) -> torch.Tensor:
        c_p, r_p = self(parent_ids)
        c_c, r_c = self(child_ids)

        dist = torch.norm(c_p - c_c, dim=-1)

        return dist - r_p + r_c
