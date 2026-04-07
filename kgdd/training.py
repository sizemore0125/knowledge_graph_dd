import torch


def test_accuracy(model, dataloader, device):
    metrics = test_binary_classification_metrics(model, dataloader, device)
    return metrics["accuracy"]

def test_binary_classification_metrics(model, dataloader, device, threshold=0.5):
    model.eval()
    total = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for edges, _, labels, _ in dataloader:
            edges = edges.to(device)
            labels = labels.to(device)

            logits = model(edges).reshape(-1)
            pred_pos = torch.sigmoid(logits) >= threshold
            label_pos = labels.reshape(-1) >= 0.5

            correct += (pred_pos == label_pos).sum().item()
            total += label_pos.numel()

            tp += (pred_pos & label_pos).sum().item()
            fp += (pred_pos & ~label_pos).sum().item()
            fn += (~pred_pos & label_pos).sum().item()

    model.train()
    accuracy = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "total": int(total),
    }
