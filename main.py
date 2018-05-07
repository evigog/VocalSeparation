import network
from constants import *
import numpy as np

def generateData():
    tm = np.random.randint(time_max-10, time_max+10)
    x = np.random.rand(batch_size, tm, feature_number)

    y = np.copy(x)

    for i in range(batch_size):
        for j in range(tm):
            for k in range(feature_number):
                if j > 1:
                    y[i, j, k] = x[i, j, k] + x[i, j-1, k] + x[i, j-2, k]

    return (x, y)

net = network.RNN_network()

X, Y = generateData()
print(X.shape)
print(Y.shape)

net.fit(generateData)
