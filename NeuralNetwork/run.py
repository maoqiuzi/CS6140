#!/usr/bin/env python

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Data import Data
from NN import NN


FILENAME = './data.mat'

def get_err_rate(Y1, Y2):
    count = 0.0
    for i in range(len(Y1)):
        if Y1[i] != Y2[i]:
            count += 1
    return count / len(Y1)

if __name__ == "__main__":
    S2 = [10, 20, 30, 50, 100]
    training_errs = []
    test_errs = []
    
    for s2 in S2:
        data = Data(FILENAME)
        model = NN(data, s2)
        model.train()
        training_errs.append(get_err_rate(data.Y_trn.flatten().tolist(), model.predict(data.X_trn)))
        test_errs.append(get_err_rate(data.Y_tst.flatten().tolist(), model.predict(data.X_tst)))

    plt.plot(S2, training_errs)
    plt.plot(S2, test_errs)
    plt.legend(["Training Error", "Test Error"])
    plt.savefig("plot")
