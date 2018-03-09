import numpy as np
from random import random
import math
import os
import utils

# -------------- PARAMETERS -------------- #
data_folder = '../data_sphere_4d/'


# ---------------------------------------- #


def gen_sphere(N, D, R=1):
    data = np.zeros((N, D))

    for i in range(N):
        dist = None
        vec = None
        while True:
            dist = 0
            vec = []
            for j in range(D):
                t = random() * 2 - 1
                dist += t * t
                vec.append(t)
            if dist <= 1:
                break

        r = math.sqrt(dist)
        vec = np.array(vec) / r * R

        data[i, :] = vec

    return data


def main():
    n_train = 10000
    n_test = 1000
    train_data = gen_sphere(n_train, 4)
    train_labels = np.zeros(n_train)
    test_data = gen_sphere(n_test, 4)
    test_labels = np.zeros(n_test)

    utils.create_dir(data_folder)
    np.save(os.path.join(data_folder, 'train_data.npy'), train_data)
    np.save(os.path.join(data_folder, 'train_labels.npy'), train_labels)
    np.save(os.path.join(data_folder, 'test_data.npy'), test_data)
    np.save(os.path.join(data_folder, 'test_labels.npy'), test_labels)


if __name__ == '__main__':
    main()
