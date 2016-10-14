# coding=utf-8

import random
import operator
import sys

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

from visualisation_util import plot_areas, plot_2d_classes

from multiprocessing import Queue

SHOW_PREDICTIONS_AND_REAL_VALUES = False


def read_data(only_2d = True):
    iris = datasets.load_iris()
    X, Y = iris.data, iris.target
    if only_2d:
        X = X[:, :2]
    return X, Y


def split_for_class(iris_set, iris_class, ratio):
	X_training = []
	Y_training = []
	X_test = []
	Y_test = []
	for i in range(len(iris_set)):
		if random.random() < ratio:
			X_training.append(np.asarray(iris_set[i][0]))
			Y_training.append(iris_class)
		else:
			X_test.append(np.asarray(iris_set[i][0]))
			Y_test.append(iris_class)

	return (X_training, Y_training, X_test, Y_test)


def split_dataset(X, Y, ratio):
    X_training = []
    Y_training = []
    X_test = []
    Y_test = []

    class_0 = filter(lambda x: x[1] == 0, ((x, y) for x, y in zip(X, Y)))
    class_1 = filter(lambda x: x[1] == 1, ((x, y) for x, y in zip(X, Y)))
    class_2 = filter(lambda x: x[1] == 2, ((x, y) for x, y in zip(X, Y)))

    X_class0_training, Y_class0_training, X_class0_test, Y_class0_test = split_for_class(class_0, 0, ratio)
    X_class1_training, Y_class1_training, X_class1_test, Y_class1_test = split_for_class(class_1, 1, ratio)
    X_class2_training, Y_class2_training, X_class2_test, Y_class2_test = split_for_class(class_2, 2, ratio)

    X_training = X_class0_training + X_class1_training + X_class2_training
    Y_training = Y_class0_training + Y_class1_training + Y_class2_training
    X_test = X_class0_test + X_class1_test + X_class2_test
    Y_test = Y_class0_test + Y_class1_test + Y_class2_test
    return np.array(X_training), np.array(Y_training), np.array(X_test), np.array(Y_test)



def kNN(X, Y, test_point, k, voting):
	distances = []
	for i in range(X.shape[0]):
		distances.append((Y[i], distance.euclidean(X[i, :], test_point)))
	distances.sort(key=operator.itemgetter(1))
	neighbours = distances[:k]
	votes = {}
	for neighbour in neighbours:
		iris_class = neighbour[0]
		weight = 1
		if voting == 'weighted':
			if neighbour[1] > 0:
				weight = 1. / neighbour[1]

		if iris_class in votes:
			votes[iris_class] += weight
		else:
			votes[iris_class] = weight
	result = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)[0]
	return result[0]


def cnn_transform(X, Y, k, voting):
    X_store = []
    Y_store = []
    training_set = Queue()
    for i in range(X.shape[0]):
        training_set.put((X[i, :], Y[i]))
    p = training_set.get()
    X_store.append(p[0])
    Y_store.append(p[1])
    n = 0
    while not training_set.empty() and n < training_set.qsize():
        n += 1
        p = training_set.get()
        if kNN(np.array(X_store), np.array(Y_store), np.array(p[0]), k, voting) == p[1]:
            training_set.put(p)
        else:
            X_store.append(p[0])
            Y_store.append(p[1])
            n = 0
    return np.array(X_store), np.array(Y_store)


def perform_section_one(voting, cnn, k):
	X, Y = read_data()
	if cnn:
		X, Y = cnn_transform(X, Y, k, voting)
	plot_areas(lambda x: kNN(X, Y, x, k, voting), 0.5, X)
	plot_2d_classes(X, Y, 'rgb')
	cnn_str = '_cnn' if cnn else ''
	plt.savefig('plots/k' + str(k) + '_' + voting + cnn_str + '.png')


def perform_section_two(voting, cnn, k):
	X, Y = read_data(False)
	accuracies = []
	cnn_percents = []
	for i in range(10):
		X_training, Y_training, X_test, Y_test = split_dataset(X, Y, 0.7)
		if cnn:
			len_before = X_training.shape[0]
			X_training, Y_training = cnn_transform(X_training, Y_training, k, voting)
			len_after = X_training.shape[0]
			cnn_percents.append(float(len_after) / float(len_before) * 100.0)
        predictions = []
        for i in range(X_test.shape[0]):
            predictions.append(kNN(X_training, Y_training, X_test[i, :], k, voting))
        if SHOW_PREDICTIONS_AND_REAL_VALUES:
            print('Prediction, actual:')
            for i in range(X_test.shape[0]):
                print(predictions[i], Y_test[i])
        correct = 0
        for i in range(len(predictions)):
            if Y_test[i] == predictions[i]:
                correct += 1
        accuracies.append(float(correct) / float(len(predictions)) * 100.0)
	print("Accuracy:", str(np.mean(accuracies)) + '%', 'StdDev:', np.std(accuracies))
	if cnn:
		print("CNN:", str(np.mean(cnn_percents)) + '%', 'StdDev:', np.std(cnn_percents))


def main():
	for voting in ['normal', 'weighted']:
		for cnn in [False, True]:
			for k in [1, 5]:
				print 'voting:', voting, ', cnn:', cnn, ', k:', k
				perform_section_one(voting, cnn, k)
				perform_section_two(voting, cnn, k)



if __name__ == "__main__":
	main()