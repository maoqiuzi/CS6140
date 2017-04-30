#!/usr/bin/env python
"A class for Spectral Clustering."

import numpy as np
from Kmeans import Kmeans


class SpectralClustering(object):
    w = None
    d = None
    clusters = None

    def train(self, data, k, sigma):
        self.x = data.T
        self.n = self.x.shape[0]
        self.k = k
        self.sigma = sigma

        self.calc_w()
        self.calc_d()
        self.l = self.d - self.w
        w, v = np.linalg.eig(self.l)
        index = w.argsort()
        h = v[:, index[0]]
        for i in xrange(1, self.k):
            h = np.vstack((h, v[:, index[i]]))
        h = h.T
        # h = v[:, w.argsort()].astype(np.float)
        normalized_h = h / np.linalg.norm(h, axis=1).reshape(self.n, 1)
        kmeans = Kmeans()
        kmeans.train(normalized_h.T, self.k, 5)
        self.clusters = kmeans.best_clusters

    def calc_w(self):
        self.w = np.zeros((self.n, self.n))
        for i in xrange(self.n):
            for j in xrange(i + 1, self.n):
                self.w[i, j] = np.exp(
                    (-np.linalg.norm(self.x[i] - self.x[j])**2) / float(self.sigma))
                self.w[j, i] = self.w[i, j]

    def calc_d(self):
        self.d = np.zeros((self.n, self.n))
        for i in xrange(self.n):
            self.d[i, i] = np.sum(self.w[i, :])
