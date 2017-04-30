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
        self.normalized_Y_trn = self._normalize_Y(mat_data['Y_trn'])
        self.X_tst = mat_data['X_tst']
        self.Y_tst = mat_data['Y_tst']
        # self.Y_tst = self._normalize_Y(mat_data['Y_tst'])

    def _normalize_Y(self, Y):
        c1 = [1, 0, 0]
        c2 = [0, 1, 0]
        c3 = [0, 0, 1]
        return np.array([c1 if y == 0 else c2 if y == 1 else c3 for y in Y])
