#!/usr/bin/env python

from Data import Data
import numpy as np
from scipy.linalg import inv
from scipy.special import expit

class LogisticRegression:
    def __init__(self, data):
        self.data = data

    def train(self):
        self.data.extend_data_phi(1)
        X = self.data.X_trn
        W = np.empty(shape = [np.shape(X)[1], 0])
        for i in range(3):
            Y = np.mat(self.data.Y_trn_coding[:, i]).T
            W = np.hstack((W, self.train_gradient_descent(X, Y)))
        self.W = W

    def train_gradient_descent(self, X, Y):
        sample_size, data_size = np.shape(X)
        threshhold = 1e-5
        alpha = 0.01
        converge = False
        count = 0
        W = np.random.rand(data_size, 1)
        while count < 10000 and not converge:
            gradient = -X.T.dot(Y - expit(X.dot(W)))
            W = W - alpha * gradient
            converge = np.linalg.norm(gradient) <= threshhold
            count += 1
        return W

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict(self.W, x))
        return np.array(predictions)

    def _predict(self, W, x):
        prediction = 0
        prob = 0
        for i in range(np.shape(W)[0]):
            cur_prob = expit(x.dot(W[:, i]))[0][0]
            if expit(x.dot(W[:, i]))[0][0] > prob:
                prob = cur_prob
                prediction = i
        return prediction

    def training_error(self):
        predictions = self.predict(self.data.X_trn)
        Y = np.squeeze(np.asarray(self.data.Y_trn))
        errors = 0
        for i in range(np.size(Y)):
            if predictions[i] != Y[i]:
                errors += 1
        training_error = errors / float(np.size(Y))
        return training_error

    def test_error(self):
        predictions = self.predict(self.data.X_tst)
        Y = np.squeeze(np.asarray(self.data.Y_tst))
        errors = 0
        for i in range(np.size(Y)):
            if predictions[i] != Y[i]:
                errors += 1
        test_error = errors / float(np.size(Y))
        return test_error
