import matplotlib.pyplot as plt
import numpy as np

from activation import sigmoid, sigmoid_dash
from input import read, save
from normalize import normalize_all, normalize


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
        if mses[-1] < 3:
            return mses, wo, wh

        sigma_o = np.dot(sigma.T, net_h)  # == > 1 * 10

        sigma_h = np.dot((np.dot(sigma, wo) * sigmoid_dash(net_h_active)).T, x)  # == > 10 * 8

        wo = wo + 0.0001 * sigma_o  # == > 1 * 10
        wh = wh + 0.0001 * sigma_h  # == > 10 * 8



