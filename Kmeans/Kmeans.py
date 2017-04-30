#!/usr/bin/env python
"A class for Kmeans."

import numpy as np
import sys


class Kmeans(object):
    last_clusters = None
    clusters = None
    last_centers = None
    centers = None
    j = None
    last_j = None
    best_j = None
    n = None
    k = None

    def __init__(self):
        
        pass

    def train(self, data, k, r):
        # init centers with k random points
        self.x = data.T
        self.k = k
        self.n = self.x.shape[0]
        self.best_clusters = None
        # loop
        for i in xrange(r):
            self.last_clusters = None
            self.clusters = None
            self.last_centers = None
            self.centers = None
            self.j = None
            init_points = np.random.choice(self.n, k, replace=False)
            self.centers = np.array([self.x[i] for i in init_points])
            while not self.converge():
                # calculate clusters
                self.calc_clusters()
                # calculate centers
                self.calc_centers()
            if self.best_j is None:
                self.best_j = self.j
                self.best_clusters = np.copy(self.clusters)
            else:
                if self.j <= self.best_j:
                    self.best_j = self.j
                    self.best_clusters = np.copy(self.clusters)

    def calc_clusters(self):
        self.last_clusters = np.copy(self.clusters)
        self.clusters = np.zeros(self.n)
        for i in xrange(self.n):
            norms = np.linalg.norm(self.x[i] - self.centers, axis=1)
            self.clusters[i] = np.argmin(norms)

    def calc_centers(self):
        self.last_centers = np.copy(self.centers)
        for i in xrange(self.centers.shape[0]):
            with np.errstate(divide='ignore', invalid='ignore'):
                self.centers[i] = np.sum(self.x[np.where(self.clusters == i)], axis=0) \
                    / len(np.where(self.clusters == i)[0])
                self.centers[~np.isfinite(self.centers)] = 0

    def converge(self):
        if self.last_centers is None or self.last_clusters is None:
            return False
        self.calc_j()
        if np.array_equal(self.last_centers, self.centers) and np.array_equal(self.last_clusters, self.clusters) and self.j == self.last_j:
            return True
        return False

    def calc_j(self):
        self.last_j = self.j
        self.j = np.sum([np.sum(np.linalg.norm(self.x[np.where(
            self.clusters == i)] - self.centers[i], axis=1)) for i in xrange(self.k)])

