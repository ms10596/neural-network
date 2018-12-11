import matplotlib.pyplot as plt
import numpy as np

from activation import sigmoid, sigmoid_dash
from input import read, save
from normalize import normalize, normalize_all


def sum_error(sigma):
    return np.power(sigma, 2).sum() / 2


def backpropagation(x, y, wo, wh):
    mses = []
    while True:
        # x == > 515 * 8

        net_h = np.dot(x, wh.T)  # == > 515 * 10

        net_h_active = sigmoid(net_h)  # == > 515 * 10

        net_o = np.dot(net_h_active, wo.T)  # == > 515 * 1

        net_o_active = net_o  # == > 515 * 1

        sigma = y - net_o_active  # == > 515 * 1

        error = sum_error(sigma)

        mses.append(error)
        if mses[-1] < 10:
            return mses, wo, wh

        sigma_o = np.dot(sigma.T, net_h)  # == > 1 * 10

        sigma_h = np.dot((np.dot(sigma, wo) * sigmoid_dash(net_h_active)).T, x)  # == > 10 * 8

        wo = wo + 0.0001 * sigma_o  # == > 1 * 10
        wh = wh + 0.0001 * sigma_h  # == > 10 * 8


if __name__ == '__main__':
    np.random.seed(1)
    m, l, n, x, y = read()
    # m ==> 8
    # l ==> 10
    # n ==> 1
    wh = np.random.uniform(low=-5, high=5, size=(l, m))  # 10 * 8
    wo = np.random.uniform(low=-5, high=5, size=(n, l))  # 1 * 10
    x = normalize_all(x)
    y = normalize(y)
    mses, wo, wh = backpropagation(x, y, wo, wh)
    save(wo, wh)
    plt.plot(mses)
    print(mses[-1])
    plt.show()
