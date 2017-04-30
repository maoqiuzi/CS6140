#!/usr/bin/env python

from sklearn import svm
from Data import Data

FILENAME = './data.mat'

def get_err_rate(Y1, Y2):
    count = 0.0
    for i in range(len(Y1)):
        if Y1[i] != Y2[i]:
            count += 1
    return count / len(Y1)

data = Data(FILENAME)
kernels = 'linear', 'poly', 'rbf', 'sigmoid'
for kernel in kernels:
    clf = svm.SVC(kernel=kernel, tol=1e-4)
    clf.fit(data.X_trn, data.Y_trn)
    print kernel
    print "Training error:", get_err_rate(data.Y_trn.flatten().tolist(), clf.predict(data.X_trn))
    print "Test error:", get_err_rate(data.Y_tst.flatten().tolist(), clf.predict(data.X_tst))

