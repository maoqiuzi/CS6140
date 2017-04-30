#!/usr/bin/env python


import scipy.io as sio
import numpy as np
from Data import Data
from scipy.linalg import inv
from LinearRegression import LinearRegression
from RidgeRegression import RidgeRegression


FILENAME = './linear_regression.mat'

def train_lr(degree):
    data = Data(FILENAME)
    model = LinearRegression(data, degree, method="CF")
    model.train()
    return {
        "tre": model.training_MSE(),
        "tse": model.test_MSE()
    }

def train_rg(degree, K, lambdas):
    data = Data(FILENAME)
    model = RidgeRegression(data, degree=degree, method="CF", lambdas=lambdas, K=K)
    model.train()
    return {
        "tre": model.training_MSE(),
        "tse": model.test_MSE(),
        "la": model.la
    }

def print_linear_regressions(degrees):
    for degree in degrees:
        data = Data(FILENAME)
        model = LinearRegression(data, degree, method="CF")
        model.train()
        print "degree = %d" % degree
        print "w: \n", model.W.T
        print "train error: ", model.training_MSE()
        print "test error: ", model.test_MSE()

def print_ridges(degrees, lambdas):
    data = Data(FILENAME)
    for degree in degrees:
        print "-" * 40
        print "degree = %d" % degree
        print "-" * 40
        for K in [2, 5, 10, np.shape(data.X_trn)[0]]:
            print "K = ", K
            data = Data(FILENAME)
            model = RidgeRegression(data, degree=degree, method="CF", lambdas=lambdas, K=K)
            model.train()
            print "lambda: ", model.la
            print "w: \n", model.W.T
            print "train error: ", model.training_MSE()
            print "test error: ", model.test_MSE()
            print "-" * 40

if __name__ == "__main__":
    print "B: Linear Regression\n", "=" * 60
    degrees = [2, 5, 10, 20]
    print_linear_regressions(degrees)
    print "=" * 60
    print "C: Logistic Regression\n"
    lambdas = np.arange(0, 50, 0.1)
    print_ridges(degrees, lambdas)

