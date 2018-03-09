import sys
from random import random
from math import pi, cos, sin, sqrt
import numpy as np


def gen_torus(N, R, r):
    data = np.zeros((N, 3))

    for i in range(N):
        while True:
            u = random() * 2 * pi
            v = random() * 2 * pi
            w = random()
            if w <= (R + r * cos(u)) / (R + r):
                break

        x = (R + r * cos(u)) * cos(v)
        y = (R + r * cos(u)) * sin(v)
        z = r * sin(u) / 5

        data[i, :] = [x, y, z]

    return data


def gen_two_tori(N, R, r):
    data = np.zeros((N, 3))

    for i in range(N):
        while True:
            u = random() * 2 * pi
            v = random() * 2 * pi
            w = random()
            if w <= (R + r * cos(u)) / (R + r):
                break

        x = (R + r * cos(u)) * cos(v)
        y = (R + r * cos(u)) * sin(v)
        z = r * sin(u)

        if random() > 0.5:
            x += 4 * R

        data[i, :] = [x, y, z]

    return data


def gen_sphere(N, R):
    data = np.zeros((N, 3))

    for i in range(N):
        while True:
            x = random() * 2 - 1
            y = random() * 2 - 1
            z = random() * 2 - 1
            if x * x + y * y + z * z <= 1:
                break

        r = sqrt(x * x + y * y + z * z)
        x = x / r * R
        y = y / r * R
        z = z / r * R

        data[i, :] = [x, y, z]

    return data


def gen_4d_sphere(N, R):
    data = np.zeros((N, 4))

    for i in range(N):
        while True:
            x = random() * 2 - 1
            y = random() * 2 - 1
            z = random() * 2 - 1
            w = random() * 2 - 1
            if x * x + y * y + z * z + w * w <= 1:
                break

        r = sqrt(x * x + y * y + z * z + w * w)
        x = x / r * R
        y = y / r * R
        z = z / r * R
        w = w / r * R

        data[i, :] = [x, y, z, w]

    return data


def gen_two_spheres(N, R, dist=4):
    data = np.zeros((N, 3))

    for i in range(N):
        while True:
            x = random() * 2 - 1
            y = random() * 2 - 1
            z = random() * 2 - 1
            if x * x + y * y + z * z <= 1:
                break

        r = sqrt(x * x + y * y + z * z)
        x = x / r * R
        y = y / r * R
        z = z / r * R
        if random() > 0.5:
            x += dist * R

        data[i, :] = [x, y, z]

    return data


def gen_4d_mobius(N, R, r):
    data = np.zeros((N, 4))

    for i in range(N):
        theta = random() * 2 * pi
        phi = random() * 2 * pi

        x = (R + r * cos(theta)) * cos(phi)
        y = (R + r * cos(theta)) * sin(phi)
        z = r * sin(theta) * cos(phi / 2)
        w = r * sin(theta) * sin(phi / 2)

        data[i, :] = [x, y, z, w]

    return data


def main():
    argv = sys.argv
    N = int(argv[1])    # number of points
    out_name = 'data.txt'   # output file
    if len(argv) > 2:
        out_name = argv[2]

    with open(out_name, 'w') as f:
        f.write('%d\n' % N)

        data = gen_torus(N, 2, 1)
        # data = gen_sphere(N, 10)
        # data = gen_two_tori(N, 10, 5)
        # data = gen_two_spheres(N, 10, 1)
        # data = gen_4d_mobius(N, 10, 5)
        # data = gen_4d_sphere(N, 10)

        # data1 = gen_torus(N / 2, 10, 5)
        # data2 = gen_sphere(N / 2, 10)
        # data2[:, 0] += 0
        # data = np.vstack((data1, data2))

        for i in range(N):
            if data.shape[1] == 3:
                f.write('%f %f %f\n' % (data[i, 0], data[i, 1], data[i, 2]))
            elif data.shape[2] == 4:
                f.write('%f %f %f %f\n' % (data[i, 0], data[i, 1], data[i, 2], data[i, 3]))
            else:
                raise Exception('Wrong dimensions')

if __name__ == '__main__':
    main()
