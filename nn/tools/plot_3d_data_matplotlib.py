import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys


def plot_data(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])

    return fig, ax


def plot_cycle(ax, filename, data, shift=0.0, color='r'):
    edges = np.loadtxt(filename, dtype=int, skiprows=1)
    if edges.shape[1] == 2:
        for i in range(edges.shape[0]):
            v1 = edges[i, 0]
            v2 = edges[i, 1]
            points = np.vstack((data[v1], data[v2])) + shift
            ax.plot(points[:, 0], points[:, 1], points[:, 2], c=color)
    else:
        print str(edges.shape[1] - 1) + "-dim cycles are not supported"

data = np.loadtxt('./data8.txt', skiprows=1)
fig, ax = plot_data(data)

sh = min(np.max(data[:, 0]) - np.min(data[:, 0]),
         np.max(data[:, 1]) - np.min(data[:, 1]),
         np.max(data[:, 2]) - np.min(data[:, 2])) * 0.01
shifts = [0, sh, -sh, 2 * sh, -2 * sh]
colors = ['r', 'g', 'purple', 'black', 'cyan']
cycles = ['repr_0.txt', 'repr_1.txt', 'repr_2.txt', 'repr_3.txt', 'repr_4.txt']  # read from arguments

for i, fname in enumerate(cycles):
    if os.path.exists(fname):
        plot_cycle(ax, fname, data, shifts[i], colors[i])

plt.show()
# plot_url = py.plot_mpl(fig)
# print plot_url
