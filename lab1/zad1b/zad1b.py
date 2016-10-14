import queue

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import warnings

warnings.filterwarnings("ignore")

def read_data(only_2_features=True):
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    if only_2_features:
        X = X[:, :2]
    return X, y


def split_dataset(X, y, ratio):
    X_training = []
    y_training = []
    X_test = []
    y_test = []
    for i in range(X.shape[0]):
        if np.random.random_sample() < ratio:
            X_training.append(X[i, :])
            y_training.append(y[i])
        else:
            X_test.append(X[i, :])
            y_test.append(y[i])
    return np.array(X_training), np.array(y_training), np.array(X_test), np.array(y_test)


def compute(X,y, n_neighbors, weights, mode):
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    if mode == "cnn":
        X, y = cnn(X, y, n_neighbors, weights)

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))
    plt.savefig( "k" + str(n_neighbors) + "_weights" + weights + "_" + mode + ".png")
    plt.clf()

def cnn(X,y, n_neighbors, weights):
    X_store = []
    y_store = []
    bag = queue.Queue()
    for i in range(X.shape[0]):
        bag.put((X[i, :], y[i]))
    p = bag.get()
    X_store.append(p[0])
    y_store.append(p[1])
    n = 0
    while not bag.empty() and n < bag.qsize():
        n += 1
        p = bag.get()
        k = n_neighbors
        X_step = np.array(X_store)
        if len(X_store) <k:
            k = len(X_store)
        clf = neighbors.KNeighborsClassifier(k, weights=weights)
        clf.fit(X_step, np.array(y_store))
        if clf.predict(np.array(p[0])) == p[1]:
            bag.put(p)
        else:
            X_store.append(p[0])
            y_store.append(p[1])
            n = 0
    return np.array(X_store), np.array(y_store)


def compute2(all_X, all_y, n_neighbors, weights,mode):
        accuracies = []
        cnn_percents = []
        for i in range(10):
            X_training, y_training, X_test, y_test = split_dataset(all_X, all_y, 0.7)

            if mode == "cnn":
                len_before = X_training.shape[0]
                X_training, y_training = cnn(X_training, y_training, n_neighbors, weights)
                len_after = X_training.shape[0]
                cnn_percents.append(float(len_after) / float(len_before) * 100.0)

            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(X_training, y_training)
            predictions = clf.predict(X_test)

            correct = 0
            for i in range(len(predictions)):
                if y_test[i] == predictions[i]:
                    correct += 1
            accuracies.append(float(correct) / float(len(predictions)) * 100.0)
        print("Accuracy:", str(np.mean(accuracies)) + '%', 'StdDev:', np.std(accuracies), "_" ,mode )
        if mode == "cnn":
            print("CNN:", str(np.mean(cnn_percents)) + '%', 'StdDev:', np.std(cnn_percents))

def main():
    for mode in ['normal', 'cnn']:
        for n_neighbors in [1, 5]:
            for weights in ['uniform', 'distance']:
                X, y = read_data()
                all_X, all_y = read_data(False)

                compute(X,y, n_neighbors, weights,mode)
                compute2(all_X, all_y, n_neighbors, weights,mode)


if __name__ == "__main__":
    main()
