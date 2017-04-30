#!/usr/bin/env python

import scipy.io as sio
import numpy as np

class Data(object):
    def __init__(self, filename):
        self.load(filename)

    def load(self, filename):
        mat_data = sio.loadmat(filename)
        self.X_trn = mat_data['X_trn']
        self.Y_trn = mat_data['Y_trn']
        self.X_tst = mat_data['X_tst']
        self.Y_tst = mat_data['Y_tst']

    def extend_data_phi(self, degree):
        self.X_trn = np.array([[x[0] ** i for i in range(degree+1)] for x in self.X_trn])
        self.X_tst = np.array([[x[0] ** i for i in range(degree+1)] for x in self.X_tst])

    def extend_data_phi_normalize(self, degree):
        self.extend_data_phi(degree)
        means, stds = self._normalize_trn()
        self._normalize_tst(means, stds)

    def _normalize_trn(self):
        X = self.X_trn
        row, col = np.shape(X)
        res = X[:, [0]]
        means = []
        stds = []
        for i in range(1, col):
            r = X[:, [i]]
            mean = np.mean(r)
            std = np.std(r)
            res = np.hstack((res, (r - mean) / std))
            means.append(mean)
            stds.append(std)
        self.X_trn = res
        return means, stds

    def _normalize_tst(self, means, stds):
        X = self.X_tst
        row, col = np.shape(X)
        res = X[:, [0]]
        for i in range(1, col):
            r = X[:, [i]]
            res = np.hstack((res, (r - means[i - 1]) / stds[i - 1]))
        self.X_tst = res


