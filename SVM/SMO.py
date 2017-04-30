 #!/usr/bin/env python

import numpy as np
import random
from Data import Data


class SMO(object):
    def __init__(self, data, C = 0.1):
        self.C = C
        self.data = data
        self.N = self.data.X_trn.shape[0]

    def train(self):
        X = self.data.X_trn
        self.Y_0_vs_all = self.data.Y_vs_all(0)
        self.Y_1_vs_all = self.data.Y_vs_all(1)
        self.Y_2_vs_all = self.data.Y_vs_all(2)
        self.A0, self.b0 = self._train(X, self.Y_0_vs_all)
        self.A1, self.b1 = self._train(X, self.Y_1_vs_all)
        self.A2, self.b2 = self._train(X, self.Y_2_vs_all)

    def predict(self, X):
        prd_0 = np.squeeze((self.A0 * self.Y_0_vs_all).T.dot(self.data.X_trn.dot(X.T)) + self.b0)
        prd_1 = np.squeeze((self.A1 * self.Y_1_vs_all).T.dot(self.data.X_trn.dot(X.T)) + self.b1)
        prd_2 = np.squeeze((self.A2 * self.Y_2_vs_all).T.dot(self.data.X_trn.dot(X.T)) + self.b2)
        results = []
        for i in range(X.shape[0]):
            if prd_0[i] > prd_1[i]:
                if prd_0[i] > prd_2[i]:
                    results.append(0)
                else:
                    results.append(2)
            else:
                if prd_1[i] > prd_2[i]:
                    results.append(1)
                else:
                    results.append(2)
        return results
        
    def _train(self, X, Y):
        C = self.C
        tol = 0.01
        max_passes = 1000
        

        A = np.zeros((self.N, 1))
        A_old = np.copy(A)
        b = 0
        passes = 0
        while passes < max_passes:
            num_changed_alphas = 0

            for i in range(self.N):
                Ei = ((A * Y).T.dot(X.dot(X[i].T)) + b - Y[i])[0]
                if (Y[i] * Ei < -tol and A[i] < C) or (Y[i] * Ei > tol and A[i] > 0):
                    j = random.choice(range(i) + range(i + 1, self.N))
                    Ej = ((A * Y).T.dot(X.dot(X[j].T)) + b - Y[j])[0]
                    A_old[i] = A[i]
                    A_old[j] = A[j]

                    # compute L and H
                    if Y[i] != Y[j]:
                        L = max(0, A[j] - A[i])
                        H = min(C, C + A[j] - A[i])
                    else:
                        L = max(0, A[i] + A[j] - C)
                        H = min(C, A[i] + A[j])
                    if L == H:
                        continue
                    eta = 2 * X[i].T.dot(X[j]) - X[i].T.dot(X[i]) - X[j].T.dot(X[j])
                    if eta >= 0:
                        continue
                    A[j] = A[j] - (Y[j] * (Ei - Ej) / eta)
                    A[j] = H if A[j] >= H else A[j] if A[j] >= L else L
                    if abs(A[j] - A_old[j]) < 0.00001:
                        continue
                    A[i] = A[i] + Y[i] * Y[j] * (A_old[j] - A[j])

                    b1 = b - Ei - Y[i] * (A[i] - A_old[i]) * X[i].T.dot(X[i]) \
                         - Y[j] * (A[j] - A_old[j]) * X[i].T.dot(X[j])
                    b2 = b - Ej - Y[i] * (A[i] - A_old[i]) * X[i].T.dot(X[j]) \
                         - Y[j] * (A[j] - A_old[j]) * X[j].T.dot(X[j])
                    if A[i] > 0 and A[i] < C:
                        b = b1
                    elif A[j] > 0 and A[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        return A, b

