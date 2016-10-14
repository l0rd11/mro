import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


def compute_insphere_ratio():
    x = []
    y = []
    y_err = []

    for n in range(2, 21):
        inside_ratios = []
        sphere_center = [1] * n

        for k in range(1, 11):
            inside = 0

            for i in range(1, 1001):
                point = np.random.random_sample((n,)) * 2
                if distance.euclidean(sphere_center, point) <= 1:
                    inside += 1

            inside_ratios.append(inside / 10.)

        x.append(n)
        y.append(np.mean(inside_ratios))
        y_err.append(np.std(inside_ratios))

    plt.ylabel('% of points inside hypersphere')
    plt.xlabel('n of dimensions')
    plt.errorbar(x, y, y_err)
    plt.savefig("insphere_ratio.png")
    plt.clf()


def compute_std_mean():
    x = []
    y = []
    y_err = []

    for n in range(2, 41):
        std_to_means = []

        for k in range(1, 11):
            hypercube = []
            distances = []

            for i in range(1, 101):
                point = np.random.random_sample((n,))
                hypercube.append(point)

            for i in range(0, len(hypercube)-1):
                for j in range(i+1, len(hypercube)):
                    dist = distance.euclidean(hypercube[i], hypercube[j])
                    distances.append(dist)

            mean = np.mean(distances)
            std = np.std(distances)
            std_to_means.append(std / mean)

        x.append(n)
        y.append(np.mean(std_to_means))
        y_err.append(np.std(std_to_means))

    plt.ylabel('std/mean for distances between points in hypercube')
    plt.xlabel('n of dimensions')
    plt.errorbar(x, y, y_err)
    plt.savefig("std_to_means.png")
    plt.clf()


def main():

    compute_insphere_ratio()

    compute_std_mean()



if __name__ == "__main__":
    main()