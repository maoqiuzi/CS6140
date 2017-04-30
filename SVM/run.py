#!/usr/bin/env python

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Data import Data
from SMO import SMO


FILENAME = './data.mat'

def get_err_rate(Y1, Y2):
    count = 0.0
    for i in range(len(Y1)):
        if Y1[i] != Y2[i]:
            count += 1
    return count / len(Y1)

if __name__ == "__main__":
    data = Data(FILENAME)
    training_errs = []
    test_errs = []
    Cs = np.arange(0, 100, 5)
    for C in Cs:
        model = SMO(data, C)
        model.train()
        training_errs.append(get_err_rate(data.Y_trn.flatten().tolist(), model.predict(data.X_trn)))
        test_errs.append(get_err_rate(data.Y_tst.flatten().tolist(), model.predict(data.X_tst)))
        print C

    plt.plot(Cs, training_errs)
    plt.plot(Cs, test_errs)
    plt.legend(["Training Error", "Test Error"])
    plt.savefig("plot2")

    

