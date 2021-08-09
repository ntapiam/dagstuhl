import torch
from isstorch import compute


def generate_examples():
    noise = torch.randn(500, 100)
    Y = torch.zeros(1000, 100)
    for k in range(99):
        Y[:500, k + 1] = 0.4 * Y[:500, k] + 0.5 + noise[:, k + 1] + 0.5 * noise[:, k]
        Y[500:, k + 1] = 0.8 * Y[500:, k] + 0.5 + noise[:, k + 1] + 0.7 * noise[:, k]

    labels = torch.cat(
        [torch.zeros(500, dtype=torch.long), torch.ones(500, dtype=torch.long)]
    )

    return (Y.unsqueeze(-1), labels)


def compute_signatures(X, level=2):
    return torch.vstack([compute(x, level) for x in X])
