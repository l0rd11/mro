# coding=utf-8
import cv2
import numpy as np
from sklearn.cluster import KMeans

if __name__ == "__main__":

    img = cv2.imread("input.png")
    shape = img.shape
    img = img.reshape(-1, img.shape[-1])  # Remove alpha
    img = np.array(img, int)

    for k in [2, 4, 8, 16, 32]:
        kmeans = KMeans(k).fit(img)
        closest_cluster = kmeans.predict(img)
        output = []
        cluster_centers = kmeans.cluster_centers_
        for x in closest_cluster:
            output.append(cluster_centers[x])
        output = np.array(output)
        cv2.imwrite("output" + str(k) + ".png", output.reshape(shape))
