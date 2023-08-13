import numpy as np

def load_data():
    X = np.load("./X.npy")
    y = np.load("./y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y

def load_weights():
    w1 = np.load("./w1.npy")
    b1 = np.load("./b1.npy")
    w2 = np.load("./w2.npy")
    b2 = np.load("./b2.npy")
    return w1, b1, w2, b2

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
