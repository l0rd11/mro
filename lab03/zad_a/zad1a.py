# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def plot_2d_classes(X, y, colors, ax=None):
    if not ax:
        ax = plt.gca()
    for i, c in zip(range(len(colors)), colors):
        idx = np.where(y == i)
        ax.scatter(X[idx, 0], X[idx, 1], c=c)


def generate_points_in_circle(n1, x, y, r, cls):
    ts = np.random.uniform(0, 2 * np.pi, n1)
    rs = np.random.uniform(0, r, n1)
    points = np.array([[rs[i] * np.cos(ts[i]) + x, rs[i] * np.sin(ts[i]) + y] for i in range(n1)])
    ys = [cls for _ in range(n1)]
    return points, ys

def generate_dataset(n=50):
    centers_xs = [-1.5, 0, 1.5]
    centers_ys = [-1.5, 0, 1.5]
    r = 0.5
    Xs = []
    ys = []
    for i in range(len(centers_xs)):
        x_center = centers_xs[i]
        for j in range(len(centers_ys)):
            y_center = centers_ys[j]
            X_i, y_i = generate_points_in_circle(n, x_center, y_center, r, i * len(centers_ys) + j)
            Xs.append(X_i)
            ys.append(y_i)
    X = Xs[0]
    y = ys[0]
    for i in range(1, len(Xs)):
        X = np.concatenate((X, Xs[i]))
        y = np.concatenate((y, ys[i])).ravel()
    return X, y


def k_means(init_method, X, k, iterations):
    means = init_method(X, k)
    qualities = []
    for i in range(iterations):
        kmeans = KMeans(n_clusters=k,max_iter=1, init=means,n_init=1).fit(X)
        new_quality = silhouette_score(X, kmeans.labels_)
        means = kmeans.cluster_centers_
        qualities.append((i, new_quality))
    return qualities


def random(X, k):
    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])
    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])
    xs = np.random.uniform(min_x, max_x, k)
    ys = np.random.uniform(min_y, max_y, k)
    return np.array([[xs[i], ys[i]] for i in range(k)])


def forgy(X, k):
    kmeans = KMeans(n_clusters=k, max_iter=1, init='random', n_init=1).fit(X)
    return kmeans.cluster_centers_


def random_partition(X, k):
    clusters = [[] for _ in range(k)]
    for x in X:
        rand = np.random.random_integers(0, k - 1)
        clusters[rand].append(x)
    return np.array([np.average(clusters[i], axis=0) for i in range(k)])


def kmeanspp(X, k):
    kmeans = KMeans(n_clusters=k, max_iter=1, init='k-means++', n_init=1).fit(X)
    return kmeans.cluster_centers_


if __name__ == '__main__':
    X, y = generate_dataset(100)
    colors = ['r', 'g', 'b', 'brown', 'c', 'm', 'y', 'k', 'pink']
    plot_2d_classes(X, y, colors)
    plt.gca().set_title("Dataset")
    plt.savefig('dataset_a.png')
    fig = plt.figure()
    ax = plt.gca()
    iterations = 50
    k = 9
    reps = 10
    for method in [random, forgy, random_partition, kmeanspp]:
        print(method)
        qualities = []
        for _ in range(reps):
            for i, quality in k_means(method, X, k, iterations):
                qualities.append((i, quality))
        avgs = []
        stds = []
        array = np.array(qualities).reshape(reps, iterations, 2)  # 2 is dim of samples
        for i in range(iterations):
            avgs.append(np.average(array[:, i, 1]))
            stds.append(np.std(array[:, i, 1]))
        ax.errorbar(range(iterations), avgs, yerr=stds, label=method.__name__[:])
    ax.legend(loc=4)
    plt.savefig('qualities_a.png')