import torch


def test_accuracy(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for edges, _, labels, _ in dataloader:
            edges = edges.to(device)
            labels = labels.to(device)

            logits = model(edges).squeeze(1)
            preds = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)

            correct += (preds == labels).sum().item()
            total += labels.numel()

    model.train()
    return correct / total if total > 0 else 0.0
