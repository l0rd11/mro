# coding=utf-8

import cv2
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def load_images(dir):
    shape = cv2.imread(dir + '0.jpg').shape
    X = np.array([cv2.imread(dir + str(i) + '.jpg').ravel() for i in range(30)], int)
    y1 = np.array([0] * 15)
    y2 = np.array([1] * 15)
    y = np.concatenate((y1, y2)).ravel()
    return X, y, shape


def scale(img, new_max):
    return (img - np.amin(img)) / float(np.amax(img) - np.amin(img)) * new_max

def mean_image(pca, shape):
    mean = pca.mean_
    img = np.array(mean)
    return scale(img, 255).reshape(shape)



def principal_components(num_of_components, pca, shape):
    selfies_output = 'selfie/principal/'
    components = pca.components_

    for x in range(num_of_components):
        img = np.array([components[x, i] for i in range(len(components[0]))])
        img = scale(img, 255)
        cv2.imwrite(selfies_output +  'principal' + str(x) + '.jpg',img.reshape(shape))


def selfies_reduced(X, num_of_components, shape):
    selfies_output = 'selfie/output' + str(num_of_components) + '/'
    pca = PCA(num_of_components).fit(X)
    mean = pca.mean_
    components = pca.components_

    Y = pca.transform(X)
    t = np.dot(Y, components)
    images = mean + t
    for (i, img) in enumerate(images):
        out_img = np.array(img)
        cv2.imwrite(
            selfies_output + 'output' + str(i) + '_' + str(num_of_components) + '.jpg'
            ,
            out_img.reshape(shape)
        )


if __name__ == '__main__':
    X, y, shape = load_images('selfie/input/')

    pca = PCA().fit(X)
    cv2.imwrite('selfie/output/mean.jpg', mean_image(pca, shape))
    transformed_X = pca.transform(X)
    plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y,
                cmap=matplotlib.colors.ListedColormap(["red", "blue"]))
    plt.title("Selfies in 2D")
    plt.savefig('selfie/output/2d.png')

    plt.figure()
    principal_components(30, pca, shape)
    for num_of_components in [5, 15, 50]:

        pca = PCA(num_of_components).fit(X)
        variance_ratio = pca.explained_variance_ratio_
        ind = np.arange(np.minimum(num_of_components,len(variance_ratio)))
        plt.bar(ind, variance_ratio)
        plt.title('Variance ratio (components: ' + str(num_of_components) + ')')
        plt.savefig('selfie/output/variance-ratio-' + str(num_of_components) + '.png')

        plt.figure()
        selfies_reduced(X, num_of_components, shape)