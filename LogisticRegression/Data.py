#!/usr/bin/env python

import numpy as np
import scipy.io as sio

class Data(object):
    def __init__(self, filename):
        self.load(filename)
    def load(self, filename):
        mat_data = sio.loadmat(filename)
        self.X_trn = np.array(mat_data['X_trn'])
        self.Y_trn = np.array(mat_data['Y_trn'])
        self.X_tst = np.array(mat_data['X_tst'])
        self.Y_tst = np.array(mat_data['Y_tst'])

    def extend_data_phi(self, degree):
        self._extend_X(degree)
        self._extend_Y()

    def _extend_X(self, degree):
        trn_size = np.shape(self.X_trn)[0]
        ones = np.ones((trn_size, 1))
        self.X_trn = np.hstack((ones, self.X_trn))
        tst_size = np.shape(self.X_tst)[0]
        ones = np.ones((tst_size, 1))
        self.X_tst = np.hstack((ones, self.X_tst))

    def _extend_Y(self):
        Y = self.Y_trn
        res = np.empty(shape = [np.shape(Y)[0], 0])
        for i in range(3):
            res = np.hstack((res, (Y == i) + 0))
        self.Y_trn_coding = res
