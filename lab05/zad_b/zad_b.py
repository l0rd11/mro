# coding=utf-8

import cv2
import numpy as np
import pywt

def load_images(dir):
    shape = cv2.imread(dir + '0.jpg').shape
    X = np.array([cv2.imread(dir + str(i) + '.jpg').ravel() for i in range(2)], int)
    return X, shape


def marge(arrays):
    new_arr = []
    for i in range(arrays[0].__len__()):
        if arrays[0][i] > arrays[1][i]:
            new_arr.append(arrays[0][i])
        else:
            new_arr.append(arrays[1][i])
    return new_arr


def task(dir,name):
    X, shape = load_images(dir + name + '/')
    arrays = []
    coeff_slices = []
    for (i, x) in enumerate(X):
        coeffs = pywt.wavedec(x, 'db14', level=3)
        arr, coeff_slice = pywt.coeffs_to_array(coeffs)
        arrays.append(arr)
        coeff_slices.append(coeff_slice)
    mer = marge(arrays)
    coeffs_from_arr = pywt.array_to_coeffs(mer, coeff_slices[0])
    cam_recon = pywt.waverecn(coeffs_from_arr, wavelet='db14')
    cv2.imwrite('out_' + name + '.jpg', cam_recon.reshape(shape))


if __name__ == '__main__':
    task('input/', 'a')
    task('input/', 'b')