import torch


class RulePenalties:
    def __init__(self, model) -> None:
        self.model = model

    @staticmethod
    def is_type(e: torch.Tensor, type_vec: torch.Tensor) -> torch.Tensor:
        cos = torch.nn.functional.cosine_similarity(e, type_vec, dim=1)
        return 1 - cos

    def domain_range_loss(
        self,
        edges: torch.Tensor,
        true_edges: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits).reshape(-1)

        e_h = self.model.entity_codebook(edges[:, 0])
        e_t = self.model.entity_codebook(edges[:, 2])
        domain_type_vec = self.model.entity_codebook(true_edges[:, 0])
        range_type_vec = self.model.entity_codebook(true_edges[:, 2])

        loss_h = self.is_type(e_h, domain_type_vec)
        loss_t = self.is_type(e_t, range_type_vec)

        return (p * (loss_h + loss_t)).mean()
        