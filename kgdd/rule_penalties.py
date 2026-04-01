import torch


class RulePenalties:
    def __init__(self, model) -> None:
        self.model = model

    def domain_range_loss(
        self,
        edges: torch.Tensor,
        true_edges: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits).reshape(-1)

        loss_h = torch.relu(self.model.entity_codebook.score(true_edges[:, 0], edges[:, 0]))
        loss_t = torch.relu(self.model.entity_codebook.score(true_edges[:, 2], edges[:, 2]))

        return (p * (loss_h + loss_t)).mean()
