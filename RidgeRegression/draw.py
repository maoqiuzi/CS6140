#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from run import *

FILENAME = './linear_regression.mat'

def plot_linear_regressions(degrees):
    tst_errs = []
    trn_errs = []
    for degree in degrees:
        data = Data(FILENAME)
        model = LinearRegression(data, degree, method="CF")
        model.train()
        tst_errs.append(model.test_MSE())
        trn_errs.append(model.training_MSE())

    caption = 'Test Errors For Linear Regression'
    plt.title(caption)
    plt.xlabel('Degree')
    plt.ylabel('MSE')
    plt.plot(degrees, tst_errs)
    plt.plot(degrees, tst_errs, 'bo')
    plt.savefig(caption + '.png')
    plt.clf()

    caption = 'Training Errors For Linear Regression'
    plt.title(caption)
    plt.xlabel('Degree')
    plt.ylabel('MSE')
    plt.plot(degrees, trn_errs)
    plt.plot(degrees, trn_errs, 'bo')
    plt.savefig(caption + '.png')
    plt.clf()

def plot_logistic_regressions(degrees, lambdas):
    data = Data(FILENAME)
    for degree in degrees:
        tst_errs = []
        trn_errs = []
        Ks = [2, 5, 10, np.shape(data.X_trn)[0]]
        for K in Ks:
            data = Data(FILENAME)
            model = RidgeRegression(data, degree=degree, method="CF", lambdas=lambdas, K=K)
            model.train()
            tst_errs.append(model.test_MSE())
            trn_errs.append(model.training_MSE())

        caption = 'Test Errors For Logistic Regression For n = %d' % degree
        plt.title(caption)
        plt.xlabel('K')
        plt.ylabel('MSE')
        plt.plot(Ks, tst_errs)
        plt.plot(Ks, tst_errs, 'bo')
        plt.savefig(caption + '.png')
        plt.clf()

        caption = 'Training Errors For Logistic Regression For n = %d' % degree
        plt.title(caption)
        plt.xlabel('K')
        plt.ylabel('MSE')
        plt.plot(Ks, trn_errs)
        plt.plot(Ks, trn_errs, 'bo')
        plt.savefig(caption + '.png')
        plt.clf()

if __name__ == "__main__":
    # linear regression
    # degrees = range(2, 20, 2)
    degrees = [2, 5, 10, 20]
    plot_linear_regressions(degrees)
    lambdas = np.arange(0, 50, 0.1)
    plot_logistic_regressions(degrees, lambdas)

