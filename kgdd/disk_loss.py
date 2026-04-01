import igraph as ig
import torch

from kgdd.model.hierarchical_embedding import DiskEmbeddings


class DiskLoss(torch.nn.Module):
    def __init__(
        self,
        disk_embeddings: DiskEmbeddings,
        entity_hierarchy: ig.Graph,
    ) -> None:
        super().__init__()

        if not entity_hierarchy.is_dag():
            raise ValueError("DiskLoss requires a DAG (child -> parent).")

        self.disk_embeddings = disk_embeddings

        ancestor_ids_by_node = [None] * entity_hierarchy.vcount()
        topo_order = entity_hierarchy.topological_sorting(mode="OUT")

        for node_id in reversed(topo_order):
            ancestor_ids = [node_id]
            for parent_id in entity_hierarchy.neighbors(node_id, mode="OUT"):
                ancestor_ids.extend(ancestor_ids_by_node[parent_id])
            ancestor_ids_by_node[node_id] = sorted(set(ancestor_ids))

        flat_ancestor_ids = []
        offsets = [0]

        for ancestor_ids in ancestor_ids_by_node:
            flat_ancestor_ids.extend(ancestor_ids)
            offsets.append(len(flat_ancestor_ids))

        self.register_buffer(
            "ancestor_ids",
            torch.tensor(flat_ancestor_ids, dtype=torch.long),
        )
        self.register_buffer(
            "offsets",
            torch.tensor(offsets, dtype=torch.long),
        )

    def _expand_ancestors(
        self,
        node_ids: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        flat_node_ids = node_ids.reshape(-1)

        starts = self.offsets[flat_node_ids]
        ends = self.offsets[flat_node_ids + 1]
        counts = ends - starts

        owner_ids = torch.repeat_interleave(
            torch.arange(flat_node_ids.shape[0], device=node_ids.device),
            counts,
        )

        ancestor_ranges = [self.ancestor_ids[s:e] for s, e in zip(starts.tolist(), ends.tolist())]
        expanded_ancestor_ids = torch.cat(ancestor_ranges)

        return expanded_ancestor_ids, owner_ids, counts

    def forward(
        self,
        entity_ids: torch.LongTensor,
        negative_ids: torch.LongTensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        ancestor_ids, owner_ids, counts = self._expand_ancestors(entity_ids)
        flat_entity_ids = entity_ids.reshape(-1)
        flat_negative_ids = negative_ids.reshape(-1)

        expanded_entity_ids = torch.repeat_interleave(flat_entity_ids, counts)
        expanded_negative_ids = torch.repeat_interleave(flat_negative_ids, counts)

        pos_score = self.disk_embeddings.score(ancestor_ids, expanded_entity_ids)
        neg_score = self.disk_embeddings.score(ancestor_ids, expanded_negative_ids)

        pair_loss = torch.relu(pos_score) + torch.relu(margin - neg_score)

        loss_by_example = torch.zeros(
            flat_entity_ids.shape[0],
            device=pair_loss.device,
            dtype=pair_loss.dtype,
        )
        loss_by_example.index_add_(0, owner_ids, pair_loss)
        loss_by_example = loss_by_example / counts.to(pair_loss.dtype)

        return loss_by_example.mean()
