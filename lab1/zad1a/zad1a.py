import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt



def generate_point_pair(n):
    point1 = np.random.random_sample((n,))*2
    point2 = np.random.random_sample((n,))*2
    return (point1, point2)


def count_distance(point_pair):
    return distance.euclidean(point_pair[0], point_pair[1])




def main():

    x = []
    y = []
    y_err = []
    wy = []
    wy_err = []
    for n in range(2, 201):
        ones = np.ones(n)
        coefficients = []
        withins = []
        for k in range(1, 10):
            distances = []
            within = 0
            for i in range(1, 1001):
                point_pair = generate_point_pair(n)
                for j in [0,1]:
                    if count_distance((ones,point_pair[j]))<=1:
                        within +=1
                distance = count_distance(point_pair)
                distances.append(distance)

            mean = np.mean(distances)
            std = np.std(distances)
            withins.append(within/20.)
            coefficients.append(std/mean)
        x.append(n)
        y.append(np.mean(coefficients))
        y_err.append(np.std(coefficients))
        wy.append(np.mean(withins))
        wy_err.append(np.std(withins))

    plt.ylabel('std / mean')
    plt.xlabel('dim')
    plt.errorbar(x, y, y_err)
    plt.savefig("stdMean.png")
    plt.clf()

    plt.ylabel('percent of points in')
    plt.xlabel('dim')
    plt.errorbar(x, wy, wy_err)
    plt.savefig("percent.png")
    plt.clf()



if __name__ == "__main__":
    main()
