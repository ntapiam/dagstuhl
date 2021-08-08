import torch


def compute(x):
    S = torch.zeros(x.shape[0], x.shape[1], x.shape[1], device=x.device)
    S[2:] = torch.einsum("ij,ik->ijk", x[1:-1] - x[0], x.diff(dim=0)[1:]).cumsum(dim=0)
    return S
