#!/usr/bin/env python
"Used to run PCA."

import matplotlib.pyplot as plt
from Data import Data
from PCA import PCA
from Kmeans import Kmeans
from SpectralClustering import SpectralClustering


FILENAME = 'data.mat'


def run_pca():
    data = Data(FILENAME)
    d = 2
    pca = PCA()
    pca.train(data.x1.T, d)

    plt.plot(pca.pc[0], pca.pc[1], 'ro')
    plt.savefig("pca")
    plt.clf()


def run_kmeans():
    data = Data(FILENAME)
    k = 4
    r = 10
    kmeans = Kmeans()
    kmeans.train(data.x23.T, k, 20)
    colors = ["green", "black", "red", "yellow"]
    for i, cluster in enumerate(kmeans.best_clusters):
        plt.scatter(data.x23[i][0], data.x23[i][1], color=colors[int(cluster)])
    # print kmeans.best_clusters
    plt.savefig("kmeans")
    plt.clf()


def run_spectral_clustering():
    data = Data(FILENAME)
    k = 4
    sigmas = [0.001, 0.01, 0.1, 1]
    for j in xrange(len(sigmas)):
        model = SpectralClustering()
        model.train(data.x23.T, k, sigmas[j])
        colors = ["green", "black", "red", "yellow"]
        for i, cluster in enumerate(model.clusters):
            plt.scatter(data.x23[i][0], data.x23[i][
                        1], color=colors[int(cluster)])
        plt.savefig("spc" + str(j))
        plt.clf()


def main():
    run_pca()
    run_kmeans()
    run_spectral_clustering()


def get_err_rate(my1, my2):
    length = len(my1)
    count = 0.0
    for y1, y2 in zip(my1, my2):
        if y1 != y2:
            count += 1
    return count / length

if __name__ == "__main__":
    main()
