import sys
import os
import numpy as np
import utils
import json
import main
import algo
import encoders
import tensorflow as tf


def main_m():
    argv = sys.argv  # label_folder representative_filename params
    label_folder = argv[1]
    repr_filename = argv[2]
    param_file = argv[3]

    data = np.loadtxt(os.path.join(label_folder, 'data.txt'), skiprows=1)
    repr = np.loadtxt(os.path.join(label_folder, repr_filename), dtype=np.int64, skiprows=1)

    vertices = np.unique(repr.flatten())
    np.savetxt('cycle_vertices.txt', vertices)

    data = data[vertices]
    np.save('cycle_data_pre.npy', data)

    labels = np.zeros(data.shape[0], dtype=np.int64)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        with open(param_file) as data_file:
            params = json.load(data_file)
        encoder = encoders.SphericalEncoder
        algo.train_encode_nn(encoder, data, labels, None, None,
                             params, **main.extract_kwargs(params))


if __name__ == '__main__':
    main_m()
