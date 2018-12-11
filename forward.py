import numpy as np

from activation import sigmoid
from input import load_weights, read
from normalize import normalize, normalize_all


def sum_error(y_hat, y):
    return np.power(y_hat - y, 2).sum() / 2


def forward_propagation(x, wh, wo):
    return np.dot(sigmoid(np.dot(x, wh.T)), wo.T)


if __name__ == '__main__':
    m, l, n, x, y = read()
    x = normalize_all(x)
    y = normalize(y)
    wo, wh = load_weights()
    y_hat = forward_propagation(x, wh, wo)
    print(sum_error(y, y_hat))
