import cv2
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def parse_data(data):
    X = []
    ys = []
    shape = data.shape
    for x in range(shape[0]):
        for y in range(shape[1]):
            if np.array_equal(data[x][y], [255, 0, 0]):
                X.append([x, y])
                ys.append(0)
            if np.array_equal(data[x][y], [0, 0, 255]):
                X.append([x, y])
                ys.append(1)
    return np.array(X),np.array(ys)



def load_dataset():
    data = cv2.imread("data.bmp")

    data = np.array(data, int)
    X, y = parse_data(data)
    return X, y


if __name__ == "__main__":
    X, y = load_dataset()
    for kernel in ['linear','rbf','poly']:
        for c in [0.0001,0.0005,0.001,0.005,0.01,0.05,1]:
            clf = SVC(kernel=kernel,C=c,degree=3)
            clf.fit(X, y)
            w = clf.dual_coef_.dot(clf.support_vectors_)
            margin = 2 / np.sqrt((w ** 2).sum())
            score = clf.score(X,y)
            print(kernel," margin ",margin," c = ",c , " score = ",score)
            plt.figure()
            plt.clf()
            plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

            # Circle out the test data
            plt.scatter(X[:, 0], X[:, 1], s=80, facecolors='none', zorder=10)

            plt.axis('tight')
            x_min = X[:, 0].min()
            x_max = X[:, 0].max()
            y_min = X[:, 1].min()
            y_max = X[:, 1].max()

            XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
            Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(XX.shape)
            plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
            plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                        levels=[-.5, 0, .5])

            plt.title(kernel)
            plt.savefig(kernel + "_" + str(c) + ".png")


