import scipy.special


def sigmoid(x):
    return scipy.special.expit(x)
    # return 1 / (1 + np.exp(-x))


def sigmoid_dash(x):
    return x * (1 - x)

