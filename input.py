import numpy as np


def read():
    f = open("train.txt", "r")
    line = list(map(int, f.readline().split()))
    m = line[0]
    l = line[1]
    n = line[2]
    k = list(map(int, f.readline().split()))[0]

    x = np.empty((k, m))
    y = np.empty((k, n))
    for i in range(k):
        line = list(map(float, f.readline().split()))
        x[i] = line[0:m]
        y[i] = line[m:]
    print(x.shape)
    print(y.shape)
    return m, l, n, x, y


if __name__ == '__main__':
   read()