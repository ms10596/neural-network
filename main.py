import numpy as np
import matplotlib.pyplot as plt

from back import backpropagation
from input import read, save
from normalize import normalize, normalize_all

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
