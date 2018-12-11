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
    return m, l, n, x, y


def save(wo, wh):
    f = open("weights.txt", "w")

    for i in wo:
        for j in i:
            f.write(str(j))
            f.write(" ")
    f.write('\n')
    for i in wh:
        for j in i:
            f.write(str(j))
            f.write(" ")


def load_weights():
    pass
    f = open("weights.txt")
    wo = np.array(list(map(float, f.readline().split())))
    wo = wo.reshape((1, 10))
    wh = np.array(list(map(float, f.readline().split())))
    wh = wh.reshape((10, 8))
    # print(wo)
    # print(wh)
    return wo, wh
