import csv

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings("ignore")


def read_wines():
    X = []
    y = []
    with open('wine.data', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            y.append(int(row[0]) - 1)
            X.append([float(row[i + 1]) for i in range(13)])
    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), y


class StackingClassifier(object):
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.fitted = False
        self.classifier = LinearRegression()

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)
        X_to_aggregation = np.array([[classifier.predict(x)[0] for classifier in self.classifiers] for x in X])
        self.classifier.fit(X_to_aggregation, y)
        self.fitted = True
        return self

    def predict(self, x):
        if not self.fitted:
            raise AssertionError('Tried to predict before fitting!')
        return [int(round(self.classifier.predict([classifier.predict(x)[0]
                                                   for classifier in self.classifiers])[0]))]

    def score(self,X_test,y_test):
        return np.sum([1 if self.predict(X_test[i, :])[0] == y_test[i] else 0
                       for i in range(len(X_test))]) / float(len(X_test))



if __name__ == '__main__':

    dataset_name = "wines"
    print('Processing dataset:', dataset_name)
    X, y = read_wines()
    X, y = shuffle(X, y, random_state=0)
    n = len(X)
    n_train = int(0.7 * n)
    X_train = X[:n_train, :]
    y_train = y[:n_train]
    X_test = X[n_train:, :]
    y_test = y[n_train:]
    classifiers = [SVC(), SVC(C=2.0), SVC(kernel='poly', degree=2), SVC(kernel='poly', degree=2, C=2.0),
                   KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=1, p=1),
                   KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=5, p=1)]
    classifiers_labels = ['SVM (C=1, kernel=rbf)', 'SVM (C=2, kernel=rbf)',
                          'SVM (C=1, kernel=polynomial; 2)', 'SVM (C=2, kernel=polynomial; 2)',
                          'kNN (k=1, metric=euclid)', 'kNN (k=1, metric=manhattan)',
                          'kNN (k=5, metric=euclid)', 'kNN (k=5, metric=manhattan)',
                          'Aggregated']
    stacking = StackingClassifier(classifiers)
    stacking.fit(X_train, y_train)
    accuracies = [classifier.score( X_test, y_test) for classifier in classifiers]
    accuracies.append(stacking.score(X_test, y_test))
    fig = plt.figure(figsize=(12, 12))
    ax = plt.gca()
    x = [i + .5 for i in range(len(classifiers_labels))]
    ax.bar(x, accuracies, width=.5)
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers_labels, rotation=45, fontsize=8)
    ax.set_ylabel('accuracy')
    plt.tight_layout()
    plt.savefig('B_' + dataset_name + '_accuracy.png', dpi=100)
