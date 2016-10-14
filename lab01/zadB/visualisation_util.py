import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


def generate_evil_2d_set():
    X1, y1 = datasets.make_gaussian_quantiles(cov=2.,
                                              n_samples=200, n_features=2,
                                              n_classes=2, random_state=1)
    X2, y2 = datasets.make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                              n_samples=300, n_features=2,
                                              n_classes=2, random_state=1)
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, - y2 + 1))
    return X, y


def plot_2d_classes(X, y, colors):
    for i, c in zip(range(len(colors)), colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=c)


def plot_areas(predict, plot_step, X):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    Z = np.array([predict(x) for x in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Spectral)


if __name__ == '__main__':
    X, y = generate_evil_2d_set()

    def predict(sample):
        import random
        return random.randint(0,1)


    plot_areas(predict, 0.1, X)
    plot_2d_classes(X, y, 'rb')

    plt.show()

