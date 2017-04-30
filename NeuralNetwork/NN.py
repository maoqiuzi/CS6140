#!/usr/bin/env python

import numpy as np
from Data import Data
from numpy import linalg as LA
from scipy.special import expit

class NN(object):
    def __init__(self, data, s2):
        self.data = data
        self.s1 = self.data.X_trn.shape[1]
        self.s2 = s2
        self.s3 = 3
        self.N = self.data.X_trn.shape[0]

    def train(self):
        X = self.data.X_trn
        Y = self.data.normalized_Y_trn
        lamb = 0.01
        alpha = 1
        # initialize weights
        w1 = np.random.normal(0, 0.0001, (self.s2, self.s1))
        w2 = np.random.normal(0, 0.0001, (self.s3, self.s2))

        b1 = np.random.normal(0, 1, (self.s2, 1))
        b2 = np.random.normal(0, 1, (self.s3, 1))

        # for loop
        for pas in range(10000):
            # forward propagation
            Z_2 = (w1.dot(X.T) + b1).T
            a2 = expit(Z_2)
            Z_3 = (w2.dot(a2.T) + b2).T
            a3 = expit(Z_3)
            # back propagation
            fp_Z3 = (1 - a3) * a3
            Delta_3 = (a3 - Y) * fp_Z3
            fp_Z2 = (1 - a2) * a2
            Delta_2 = Delta_3.dot(w2) * fp_Z2

            gradient_w2 = Delta_3.T.dot(a2)
            gradient_w1 = Delta_2.T.dot(X)
            gradient_b2 = np.sum(Delta_3.T, axis=1, keepdims=True)
            gradient_b1 = np.sum(Delta_2.T, axis=1, keepdims=True)
            w1 = w1 - alpha * (1.0 / self.N * gradient_w1 + lamb * w1)
            w2 = w2 - alpha * (1.0 / self.N * gradient_w2 + lamb * w2)
            b1 = b1 - alpha * (1.0 / self.N * gradient_b1)
            b2 = b2 - alpha * (1.0 / self.N * gradient_b2)

            if (LA.norm(gradient_w1 / self.N) < 0.001 and LA.norm(gradient_w2 / self.N) < 0.001):
                # converged
                break
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2

    def predict(self, X):
        Z_2 = (self.w1.dot(X.T) + self.b1).T
        a2 = expit(Z_2)
        Z_3 = (self.w2.dot(a2.T) + self.b2).T
        a3 = expit(Z_3)
        results = []
        for a in a3:
            if a[0] > a[1]:
                if a[0] > a[2]:
                    results.append(0)
                else:
                    results.append(2)
            else:
                if a[1] > a[2]:
                    results.append(1)
                else:
                    results.append(2)
        return results

