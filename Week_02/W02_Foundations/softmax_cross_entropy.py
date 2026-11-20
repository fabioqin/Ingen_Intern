import torch


def stable_softmax(logits, dim=-1):
    if logits.ndim == 0:
        raise ValueError("Softmax requires a tensor with at least one dimension.")
    shifted = logits - logits.amax(dim=dim, keepdim=True)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def cross_entropy(logits, targets):
    if logits.ndim != 2:
        raise ValueError("Logits must have shape (batch_size, num_classes).")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("Targets must have shape (batch_size,).")
    log_probabilities = torch.log_softmax(logits, dim=1)
    selected = log_probabilities.gather(1, targets.long().unsqueeze(1))
    return -selected.mean()


if __name__ == "__main__":
    scores = torch.tensor([[2.0, 1.0, 0.1], [0.2, 0.3, 2.0]])
    labels = torch.tensor([0, 2])
    print(stable_softmax(scores))
    print(cross_entropy(scores, labels))
