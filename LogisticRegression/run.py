#!/usr/bin/env python


import scipy.io as sio
import numpy as np
from Data import Data
from scipy.linalg import inv
from LogisticRegression import LogisticRegression

FILENAME = './logistic_regression.mat'

if __name__ == "__main__":
    data = Data(FILENAME)
    model = LogisticRegression(data)
    model.train()
    print "W: \n", model.W
    print "training error: ", model.training_error()
    print "test error: ", model.test_error()
