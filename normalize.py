def normalize(x):
    return (x - x.min()) / x.ptp()


def normalize_all(x):
    for i in range(8):
        x[:, i] = normalize(x[:, i])
    return x
