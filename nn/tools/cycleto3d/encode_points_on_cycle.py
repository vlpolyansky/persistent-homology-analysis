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
    argv = sys.argv  # params
    param_file = argv[1]

    data = np.load('cycle_data_pre.npy')

    labels = np.zeros(data.shape[0], dtype=np.int64)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        with open(param_file) as data_file:
            params = json.load(data_file)
        encoder = encoders.SphericalEncoder
        encode = algo.encoder_wrapper(encoder, params, **main.extract_kwargs(params))

        outputs = []
        step = 500
        for i in range(0, data.shape[0], step):
            outputs.append(encode(data[i: i + step], labels[i: i + step]))
        output = np.concatenate(outputs)
        np.save('cycle_data_post.npy', output)


if __name__ == '__main__':
    main_m()
