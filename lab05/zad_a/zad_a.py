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


def divide(img, param):
    c2a3 = img[0:param[0]]
    c2d3 = img[param[0]:param[0]+param[1]]
    c2d2 = img[param[0] + param[1] : param[0] + param[1] +param[2]]
    c2d1 = img[param[0] + param[1] + param[2] : param[0] + param[1] + param[2] +param[3]]
    cd3 = img[param[0] + param[1] + param[2] + param[3] :param[0] + param[1] + param[2] + param[3] + param[4]]
    cd2 = img[param[0] + param[1] + param[2] + param[3] + param[4]:param[0] + param[1] + param[2] + param[3] + param[4] + param[5]]
    cd1 = img[param[0] + param[1] + param[2] + param[3] + param[4] + param[5]:param[0] + param[1] + param[2] + param[3] + param[4] + param[5] + param[6]]
    return c2a3, c2d3, c2d2, c2d1, cd3, cd2, cd1


def mean_image(pca, shape,coeff_slices,wave):
    mean = pca.mean_
    img = np.array(mean)
    # coeffs_from_arr = pywt.array_to_coeffs(img, coeff_slices[1])

    c2a3, c2d3, c2d2, c2d1, cd3, cd2, cd1 = divide(img,coeff_slices[1])

    cf = [c2a3, c2d3, c2d2, c2d1]
    cd = pywt.waverec(cf, wavelet=wave)
    cf2 = [cd, cd3, cd2, cd1]
    cam_recon = pywt.waverec(cf2, wavelet=wave)


    # cam_recon = pywt.waverecn(cam_recon, wavelet='db14')
    return scale(cam_recon, 255).reshape(shape)



def principal_components(num_of_components, pca, shape,coeff_slices,wave):
    selfies_output = 'selfie/principal/'
    components = pca.components_

    for x in range(num_of_components):
        img = np.array([components[x, i] for i in range(len(components[0]))])
        # coeffs_from_arr = pywt.array_to_coeffs(img, coeff_slices[x])
        # cam_recon = pywt.waverecn(coeffs_from_arr, wavelet='db14')


        c2a3, c2d3, c2d2, c2d1, cd3, cd2, cd1 = divide(img, coeff_slices[1])

        cf = [c2a3, c2d3, c2d2, c2d1]
        cd = pywt.waverec(cf, wavelet=wave)
        cf2 = [cd, cd3, cd2, cd1]
        cam_recon = pywt.waverec(cf2, wavelet=wave)

        img = scale(cam_recon, 255)
        cv2.imwrite(selfies_output + wave +  'principal' + str(x) + '.jpg',img.reshape(shape))


def selfies_reduced(arrays,coeff_slices, num_of_components, shape,wave):
    selfies_output = 'selfie/output' + str(num_of_components) + '/'


    pca = PCA(num_of_components).fit(arrays)
    mean = pca.mean_
    components = pca.components_


    Y = pca.transform(arrays)
    t = np.dot(Y, components)

    images = mean + t
    for (i, img) in enumerate(images):

        # coeffs_from_arr = pywt.array_to_coeffs(img, coeff_slices[i])
        # cam_recon = pywt.waverecn(coeffs_from_arr, wavelet='db14')
        c2a3, c2d3, c2d2, c2d1, cd3, cd2, cd1 = divide(img, coeff_slices[i])

        cf = [c2a3, c2d3, c2d2, c2d1]
        cd = pywt.waverec(cf, wavelet=wave)
        cf2 = [cd, cd3, cd2, cd1]
        cam_recon = pywt.waverec(cf2, wavelet=wave)
        out_img = np.array(cam_recon)

        cv2.imwrite(
            selfies_output + wave + 'output' + str(i) + '_' + str(num_of_components) + '.jpg'
            ,
            out_img.reshape(shape)
        )




if __name__ == '__main__':
    X, y, shape = load_images('selfie/input/')

    for wave in ["db14","coif2"]:
        arrays = []
        coeff_slices = []
        for (i, x) in enumerate(X):
            ca3 ,cd3,cd2,cd1 = pywt.wavedec(x, wave, level=3)
            c2a3, c2d3, c2d2, c2d1= pywt.wavedec(ca3, wave, level=3)
            coeffs = [c2a3, c2d3, c2d2, c2d1,cd3,cd2,cd1]
            # coeffs = pywt.wavedecn(x, wavelet='db14', level=3)
            # arr, coeff_slice = pywt.coeffs_to_array(coeffs)
            slice = [len(c2a3), len(c2d3), len(c2d2), len(c2d1),len(cd3),len(cd2),len(cd1)]
            arrays.append(np.concatenate((c2a3, c2d3, c2d2, c2d1,cd3,cd2,cd1)))

            coeff_slices.append(slice)




        pca = PCA().fit(arrays)
        cv2.imwrite('selfie/output/'+wave+'mean.jpg', mean_image(pca, shape,coeff_slices,wave))
        transformed_X = pca.transform(arrays)
        plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y,
                    cmap=matplotlib.colors.ListedColormap(["red", "blue"]))
        plt.title("Selfies in 2D")
        plt.savefig('selfie/output/' +wave +'2d.png')

        plt.figure()
        principal_components(30, pca, shape,coeff_slices,wave)
        for num_of_components in [5, 15, 50]:

            pca = PCA(num_of_components).fit(arrays)
            variance_ratio = pca.explained_variance_ratio_
            ind = np.arange(np.minimum(num_of_components,len(variance_ratio)))
            plt.bar(ind, variance_ratio)
            plt.title('Variance ratio (components: ' +wave + str(num_of_components) + ')')
            plt.savefig('selfie/output/variance-ratio-'+wave + str(num_of_components) + '.png')

            plt.figure()
            selfies_reduced(arrays,coeff_slices, num_of_components, shape,wave)