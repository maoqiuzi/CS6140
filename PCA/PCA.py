#!/usr/bin/env python
"A class for PCA."

import numpy as np


class PCA(object):
    pc = None

    def __init__(self):
        pass

    def center(self, x):
        mean = x.mean(axis=0)
        return x - mean

    def train(self, data, d):
        centered_x = self.center(data.T)
        u, s, v = np.linalg.svd(centered_x)
        u = u[:, :d]
        self.pc = u.dot(np.diag(s)[:d, :d]).T
