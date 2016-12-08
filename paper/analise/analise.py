import scipy.io.wavfile as wav
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings("ignore")

import pywt


def load_data(dir):
    rate, data = wav.read(dir + '0out.wav')
    X = [wav.read(dir + str(i) + 'out.wav') for i in range(21)]
    y1 = np.array([0] * 10)
    y2 = np.array([1] * 10)
    y = np.concatenate((y1, y2)).ravel()
    plt.plot(X[1][1])
    plt.savefig('ja')
    plt.clf()
    plt.plot(X[0][1])
    plt.savefig('tymo')
    plt.clf()

    return X, y, rate


if __name__ == '__main__':
    X, y, shape = load_data('../record/')
    arrays = []
    coeff_slices = []
    for (i, x) in enumerate(X):
        norm = normalize(x[1][:, np.newaxis], axis=0).ravel()
        coeffs = pywt.wavedec(norm, 'sym13', level=2)
        arr, coeff_slice = pywt.coeffs_to_array(coeffs)
        arrays.append(arr)
        coeff_slices.append(coeff_slice)

    plt.plot(arrays[1])
    plt.savefig('ja_dwt')
    plt.clf()
    plt.plot(arrays[0])
    plt.savefig('tymo_dwt')
    plt.clf()
    pca = KernelPCA(kernel='sigmoid').fit(arrays)
    transformed_X = pca.transform(arrays)
    plt.plot(transformed_X[1])
    plt.savefig('ja_pca')
    plt.clf()
    plt.plot(transformed_X[0])
    plt.savefig('tymo_pca')
    plt.clf()
    plt.scatter(transformed_X[1:21][:, 0], transformed_X[1:21][:, 1], c=y,
                cmap=matplotlib.colors.ListedColormap(["red", "blue"]))
    # plt.scatter(transformed_X[0][:, 0], transformed_X[0][:, 1])
    plt.title("2D")
    plt.savefig('2d.png')

    clf = SVC()
    clf.fit(transformed_X[1:21], y)

    print(clf.predict(transformed_X[0]))
