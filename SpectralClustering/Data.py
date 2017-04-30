#!/usr/bin/env python
"A class for reading data."

import scipy.io as sio
import numpy as np


class Data(object):

    def __init__(self, filename):
        self.load(filename)

    def load(self, filename):
        mat_data = sio.loadmat(filename)
        self.x1 = mat_data['X_Question1'].T
        self.x23 = mat_data['X_Question2_3'].T


def main():
    data = Data("../data.mat")


if __name__ == '__main__':
    main()
