import torch


def compute(x, level=2):
    if x.dim() != 2:
        raise ValueError("Input tensor must have shape (N, d)")

    N, d  = x.shape
    dX = [x.diff(dim=0)]
    for k in range(1, level):
        dX.append(torch.einsum("t...,tj->t...j", dX[k - 1], x.diff(dim=0)))

    S = [x[1:] - x[0]] 
    for n in range(1, level):
        temp = torch.zeros((N - 1,) + (d,) * (n+1))
        for j in range(n):
            S_shape = (N - 1,) + (d,) * (j + 1) + (1,) * (n - j)
            X_shape = (N - 1,) + (1,) * (j + 1) + (d,) * (n - j)
            temp += (S[j].view(S_shape) * dX[n - j - 1].view(X_shape))
        S.append(temp.cumsum(dim=0))

    return torch.cat([s[-1].flatten() for s in S[1:]] + [dx.sum(dim=0).flatten() for dx in dX])

if __name__ == "__main__":
    x = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [2., 2.]])

