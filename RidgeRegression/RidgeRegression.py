#!/usr/bin/env python

import scipy.io as sio
import numpy as np
from Data import Data
from scipy.linalg import inv


class RidgeRegression(object):
    def __init__(self, data, degree=1, method="CF", lambdas=np.arange(0, 50), K=2):
        self.data = data
        self.degree = degree
        self.method = method
        self.lambdas = lambdas
        self.K = K

    def train(self):
        self.data.extend_data_phi_normalize(self.degree)
        X = self.data.X_trn
        Y = self.data.Y_trn
        mses = []
        for la in self.lambdas:
            mses.append(self.train_n_fold(X, Y, self.K, la))
        la = self.lambdas[mses.index(min(mses))]
        self.W = self.train_closed_form(X, Y, la)
        self.la = la

    def train_n_fold(self, X, Y, K, la):
        row, col = np.shape(X)
        span = row / K
        mses = []
        for i in range(K):
            X_holdout = X[i * span : (i + 1) * span, :]
            X_trn = np.vstack((X[0 : i * span, : ], X[(i + 1) * span : , : ]))
            Y_holdout = Y[i * span : (i + 1) * span, :]
            Y_trn = np.vstack((Y[0 : i * span, : ], Y[(i + 1) * span : , : ]))
            W = self.train_closed_form(X_trn, Y_trn, la)
            mse = self._MSE(W, X_holdout, Y_holdout)
            mses.append(mse)
        mse_mean = np.mean(mses)
        return mse_mean

    def train_closed_form(self, X, Y, la):
        col = np.shape(X)[1]
        W = inv((X.T.dot(X)) + la * np.identity(col)).dot(X.T).dot(Y)
        return W

    def _MSE(self, W, X, Y):
        return ((Y - X.dot(W)) ** 2).mean()

    def training_MSE(self):
        return self._MSE(self.W, self.data.X_trn, self.data.Y_trn)

    def test_MSE(self):
        return self._MSE(self.W, self.data.X_tst, self.data.Y_tst)
