import matplotlib
import scipy.io.wavfile as wav
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from sklearn.svm import SVC

import pywt


def load_data(dir):
    rate, data = wav.read(dir + '0out.wav')
    X = [wav.read(dir +  str(i) + 'out.wav') for i in range(7)]
    y1 = np.array([0] * 3)
    y2 = np.array([1] * 3)
    y = np.concatenate((y1, y2)).ravel()
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


    pca = KernelPCA(kernel='sigmoid').fit(arrays)
    transformed_X = pca.transform(arrays)

    plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(["red", "blue"]))
    plt.title("2D")
    plt.savefig('2d.png')

    clf = SVC()
    clf.fit(transformed_X[0:6],y)

    print(clf.predict(transformed_X[6]))
