import numpy as np


def compute(x):
    S = np.zeros((x.shape[0], x.shape[1], x.shape[1]))

    S[2:] = np.einsum("ti,tj->tij", x[1:-1] - x[0], np.diff(x, axis=0)[1:]).cumsum(axis=0)

    return S
