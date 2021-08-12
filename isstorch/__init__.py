import torch
from itertools import chain
from math import comb


def __gen_outer(x, y):
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input tensors must have the same length along dimension 0")

    x_shape = x.shape + (1,) * (y.dim() - 1)
    y_shape = (y.shape[0],) + (1,) * (x.dim() - 1) + y.shape[1:]
    return x.view(x_shape) * y.view(y_shape)


def compute(x, level=2):
    if x.dim() != 2:
        raise ValueError("Input tensor must have shape (N, d)")

    N, d = x.shape
    dX = [x.diff(dim=0)]
    for k in range(1, level):
        dX.append(torch.einsum("t...,tj->t...j", dX[k - 1], dX[0]))

    S = [[dx.cumsum(dim=0) for dx in dX]]

    for length in range(1, level):
        temp = []
        for s in S[length - 1]:
            weight = s.dim() - 1
            for x_weight in range(level - weight):
                out = torch.zeros((N - 1,) + (d,) * (weight + x_weight + 1))
                torch.cumsum(
                    __gen_outer(s[length-1:-1], dX[x_weight][length:]),
                    dim=0,
                    out=out[length:],
                )
                temp.append(out)
        S.append(temp)

    print(S)
    print([[s.shape for s in sub] for sub in S])
    return torch.cat([s[-1].flatten() for sub in S for s in sub])


if __name__ == "__main__":
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    print(compute(x, 4))
