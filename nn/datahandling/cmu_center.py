import numpy as np
from utils import *
import os
import scipy.misc
import matplotlib.pyplot as plt

# -------------- PARAMETERS -------------- #
data_folder = '../data_cmu/'
result_folder = '../data_cmu_centered/'

output_size = (64, 64)
threshold = 1


# ---------------------------------------- #


def main():
    data = np.load(os.path.join(data_folder, 'frames.npy'))
    result = np.zeros((data.shape[0], output_size[0], output_size[1], data.shape[3]), dtype=data.dtype)
    for i in range(data.shape[0]):
        frame = data[i]

        # center = center_of_mass(np.sum(frame, axis=-1))
        # frame = np.lib.pad(frame, [(output_size[0], output_size[0]), (output_size[1], output_size[1]), (0, 0)],
        # 'constant', constant_values=0)
        # l = int(center[0] - output_size[0] / 2) + output_size[0]
        # u = int(center[1] - output_size[1] / 2) + output_size[1]
        # frame = frame[l: l + output_size[0], u: u + output_size[1]]

        tmp = np.sum(frame, axis=-1)
        xs = np.argwhere(np.sum(tmp, axis=1) >= threshold)
        ys = np.argwhere(np.sum(tmp, axis=0) >= threshold)
        if xs.size < 2 or ys.size < 2:
            xs = np.array([[0], [1]])
            ys = np.array([[0], [1]])
        xs = np.array([xs[0][0], xs[-1][0]])
        ys = np.array([ys[0][0], ys[-1][0]])
        frame_shape = frame.shape
        frame = np.lib.pad(frame, [(frame.shape[0], frame.shape[0]), (frame.shape[1], frame.shape[1]), (0, 0)],
                           'constant', constant_values=0)
        xs += frame_shape[0]
        ys += frame_shape[1]
        w, h = xs[1] - xs[0], ys[1] - ys[0]
        if w < h:
            diff = int((h - w) / 2)
            xs[0] -= diff
            xs[1] += diff
            if (h - w) % 2 != 0:
                xs[1] += 1
        else:
            diff = int((w - h) / 2)
            ys[0] -= diff
            ys[1] += diff
            if (w - h) % 2 != 0:
                ys[1] += 1

        frame = frame[xs[0]: xs[1], ys[0]: ys[1]]
        frame = scipy.misc.imresize(frame, output_size)
        result[i] = frame

    create_dir(result_folder)
    np.save(os.path.join(result_folder, 'frames.npy'), result)


if __name__ == '__main__':
    main()
