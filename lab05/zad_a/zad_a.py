# coding=utf-8

import cv2
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pywt


def load_images(dir):
    shape = cv2.imread(dir + '0.jpg').shape
    X = np.array([cv2.imread(dir + str(i) + '.jpg').ravel() for i in range(30)], int)
    y1 = np.array([0] * 15)
    y2 = np.array([1] * 15)
    y = np.concatenate((y1, y2)).ravel()
    return X, y, shape


def scale(img, new_max):
    return (img - np.amin(img)) / float(np.amax(img) - np.amin(img)) * new_max

def mean_image(pca, shape,coeff_slices):
    mean = pca.mean_
    img = np.array(mean)
    coeffs_from_arr = pywt.array_to_coeffs(img, coeff_slices[1])
    cam_recon = pywt.waverecn(coeffs_from_arr, wavelet='db14')
    return scale(cam_recon, 255).reshape(shape)



def principal_components(num_of_components, pca, shape,coeff_slices):
    selfies_output = 'selfie/principal/'
    components = pca.components_

    for x in range(num_of_components):
        img = np.array([components[x, i] for i in range(len(components[0]))])
        coeffs_from_arr = pywt.array_to_coeffs(img, coeff_slices[x])
        cam_recon = pywt.waverecn(coeffs_from_arr, wavelet='db14')
        img = scale(cam_recon, 255)
        cv2.imwrite(selfies_output +  'principal' + str(x) + '.jpg',img.reshape(shape))


def selfies_reduced(arrays,coeff_slices, num_of_components, shape):
    selfies_output = 'selfie/output' + str(num_of_components) + '/'


    pca = PCA(num_of_components).fit(arrays)
    mean = pca.mean_
    components = pca.components_


    Y = pca.transform(arrays)
    t = np.dot(Y, components)

    images = mean + t
    for (i, img) in enumerate(images):

        coeffs_from_arr = pywt.array_to_coeffs(img, coeff_slices[i])
        cam_recon = pywt.waverecn(coeffs_from_arr, wavelet='db14')
        out_img = np.array(cam_recon)

        cv2.imwrite(
            selfies_output + 'output' + str(i) + '_' + str(num_of_components) + '.jpg'
            ,
            out_img.reshape(shape)
        )


if __name__ == '__main__':
    X, y, shape = load_images('selfie/input/')
    arrays = []
    coeff_slices = []
    for (i, x) in enumerate(X):
        coeffs = pywt.wavedec(x, 'db14', level=2)
        arr, coeff_slice = pywt.coeffs_to_array(coeffs)
        arrays.append(arr)
        coeff_slices.append(coeff_slice)



    pca = PCA().fit(arrays)
    cv2.imwrite('selfie/output/mean.jpg', mean_image(pca, shape,coeff_slices))
    transformed_X = pca.transform(arrays)
    plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y,
                cmap=matplotlib.colors.ListedColormap(["red", "blue"]))
    plt.title("Selfies in 2D")
    plt.savefig('selfie/output/2d.png')

    plt.figure()
    principal_components(30, pca, shape,coeff_slices)
    for num_of_components in [5, 15, 50]:

        pca = PCA(num_of_components).fit(arrays)
        variance_ratio = pca.explained_variance_ratio_
        ind = np.arange(np.minimum(num_of_components,len(variance_ratio)))
        plt.bar(ind, variance_ratio)
        plt.title('Variance ratio (components: ' + str(num_of_components) + ')')
        plt.savefig('selfie/output/variance-ratio-' + str(num_of_components) + '.png')

        plt.figure()
        selfies_reduced(arrays,coeff_slices, num_of_components, shape)