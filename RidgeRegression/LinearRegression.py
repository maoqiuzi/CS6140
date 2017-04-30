#!/usr/bin/env python

from Data import Data
import numpy as np
from scipy.linalg import inv

class LinearRegression(object):
    def __init__(self, data, degree=1, method="CF"):
        self.data = data
        self.degree = degree
        self.method = method

    def train(self):
        if self.method is "GD":
            self.data.extend_data_phi(self.degree)
            self.train_gradient_descent()
        else:
            self.data.extend_data_phi(self.degree)
            self.train_closed_form()

    def train_closed_form(self):
        X = self.data.X_trn
        Y = self.data.Y_trn
        self.W = inv((X.T.dot(X))).dot(X.T).dot(Y)

    def train_gradient_descent(self):
        X = self.data.X_trn
        Y = self.data.Y_trn
        data_size = np.shape(X)[1]
        sample_size = np.shape(X)[0]
        threshhold = 1e-7
        ro = 0.01
        converge = False
        count = 0
        W = np.mat(np.random.rand(data_size,1))

        while count < 10000 and not converge:
            gradient = 2 * X.T.dot(X.dot(W) - Y) / sample_size
            W = W - ro * gradient
            converge = np.linalg.norm(gradient) <= threshhold
            count += 1
        self.W = np.array(W)

    def _MSE(self, W, X, Y):
        return ((Y - X.dot(W)) ** 2).mean()

    def training_MSE(self):
        return self._MSE(self.W, self.data.X_trn, self.data.Y_trn)

    def test_MSE(self):
        return self._MSE(self.W, self.data.X_tst, self.data.Y_tst)
