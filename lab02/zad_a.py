# coding=utf-8

from sklearn.decomposition import PCA, KernelPCA
import numpy as np
from matplotlib import pyplot as plt


def generate_circle(n, r_min,r_max):
    n = int(n)
    angels = np.random.random_sample(n) * 2 * np.pi
    radius = np.random.random_sample(n) * (r_max - r_min) + r_min
    points = np.array([[radius[i] * np.cos(angels[i]), radius[i] * np.sin(angels[i])] for i in range(n)])
    return points


def generate_dataset_circles(n):
    X1 = generate_circle(4 * n / 9, 0, 0.4)
    y1 = [0] * int(4 * n / 9)
    X2 = generate_circle(5 * n / 9, 0.4, 1)
    y2 = [1] * int(5 * n / 9)
    return np.concatenate((X1, X2)), np.concatenate((y1, y2)).ravel()


def generate_cross_line_dataset(n):
    xs = np.random.random_sample(n) * 2 - 1
    drags = np.random.random_sample(n) * 0.2 - 0.1
    n_over_2 = int(n/2)
    X1 = np.array([[xs[i], 0.5 * xs[i] + drags[i]] for i in range(n_over_2)])
    X2 = np.array([[xs[i], 0.5 * -xs[i] + drags[i]] for i in range(n_over_2, n)])
    X = np.concatenate((X1, X2))
    y1 = [0]  * n_over_2
    y2 = [1] * n_over_2
    y = np.concatenate((y1, y2)).ravel()
    return X, y



def process(X, y, ax, pca=False, kernel=None, gamma=None, description=''):
    if pca:
        kwargs = {'gamma': gamma} if gamma else {}
        pca_obj = KernelPCA(2, kernel=kernel, **kwargs) if kernel else PCA(2)
        X_transformed = pca_obj.fit_transform(X)
        ax.set_title(description)
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        colors = {0: 'r', 1: 'b'}
        for i in range(2):
            idx = np.where(y == i)
            ax.scatter(X_transformed[idx, 0], X_transformed[idx, 1], c=colors[i])
    else:
        pca_obj = PCA(2)
        ax.set_title(description)
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        colors = {0: 'r', 1: 'b'}
        for i in range(2):
            idx = np.where(y == i)
            ax.scatter(X[idx, 0], X[idx, 1], c=colors[i])
        pca_obj = pca_obj.fit(X)
        components = pca_obj.components_
        var = pca_obj.explained_variance_ratio_
        ax.quiver(0, 0, components[0][0] * var[0], components[0][1] * var[0] ,angles='xy',scale_units='xy',scale=1)
        ax.quiver(0, 0, components[1][0] * var[1], components[1][1] * var[1] ,angles='xy',scale_units='xy',scale=1)


if __name__ == '__main__':
    X1, y1 = generate_dataset_circles(500)
    fig, (row1, row2) = plt.subplots(2, 4, figsize=(15, 15), dpi=80)
    (ax1, ax2, ax3, ax4) = row1
    process(X1, y1, ax1, description='no PCA')
    process(X1, y1, ax2, pca=True, description='PCA')
    process(X1, y1, ax3, pca=True, kernel='cosine', description='cosine PCA')
    process(X1, y1, ax4, pca=True, kernel='rbf', gamma=15, description='rbf PCA gamma 15')
    X2, y2 = generate_cross_line_dataset(500)
    (ax1, ax2, ax3, ax4) = row2
    process(X2, y2, ax1, description='no PCA')
    process(X2, y2, ax2, pca=True, description='PCA')
    process(X2, y2, ax3, pca=True, kernel='cosine', description='cosine PCA')
    process(X2, y2, ax4, pca=True, kernel='rbf', gamma=15, description='rbf PCA gamma 15')
    plt.savefig('zad_a.png')
    plt.show()