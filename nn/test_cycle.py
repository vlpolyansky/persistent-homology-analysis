import numpy as np
import sys
import json
import data as _data
import os
import cPickle
import tensorflow as tf
import encoders
import algo

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def unpickle(filename):
    with open(filename, 'rb') as fo:
        dictionary = cPickle.load(fo)
    return dictionary


def extract_kwargs(params):
    kwargs = dict()
    choices = {
        'FCAutoencoder': {
            'fc_layer_sizes': 'fc_layers'
        },
        'SphericalEncoder': {
            'fc_layer_sizes': 'fc_layers'
        },
        'MixedAutoencoder': {
            'kernel_sizes': 'kernel_sizes',
            'filters_nums': 'filters_nums',
            'pool_sizes': 'pool_sizes',
            'fc_layer_sizes': 'fc_layers'
        }
    }
    cur_choices = choices[params['algo']]
    for k in cur_choices.keys():
        kwargs[k] = params[cur_choices[k]]

    return kwargs


def read_and_make_cifar_cycle(params, index, angles, normalize=True):
    train_images = []
    train_labels = []
    for i in range(1, 6):
        data = unpickle(os.path.join(params['raw_data_folder'], 'data_batch_%d' % i))
        train_images.append(data['data'])
        train_labels.append(data['labels'])
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)

    train_images = train_images.reshape((train_images.shape[0], 3, 32, 32)).transpose((0, 2, 3, 1))

    if 'filter_labels' in params and params['filter_labels'] is not None:
        train_images, train_labels = _data.filter_labels(train_images, train_labels, params['filter_labels'])

    train_images = train_images[index: index + 1]
    train_labels = train_labels[index: index + 1]

    train_images, train_labels = _data.rotate_images(train_images, train_labels, angles)

    if normalize:
        train_images = _data.normalize_data(train_images)

    return train_images, train_labels


def fix_im(im):
    return np.minimum(np.maximum(im + 0.5, 0.), 1.)


def handle_cycle(params, images, labels):
    images = np.concatenate([images, images[:1]])
    labels = np.concatenate([labels, labels[:1]])
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        encoder = getattr(encoders, params['algo'])

        encode = algo.encoder_with_decoder_wrapper(encoder, params, **extract_kwargs(params))

        outputs = []
        losses = []
        decoded_images = []
        step = 500
        for i in range(0, images.shape[0], step):
            im_block = images[i: i + step]
            encoded, decoded = encode(im_block, labels[i: i + step])
            outputs.append(encoded)
            decoded_images.append(decoded)
            for j in range(im_block.shape[0]):
                print np.linalg.norm(decoded[j] - im_block[j])
                losses.append(np.linalg.norm(decoded[j] - im_block[j]))
        # outputs.append(outputs[0][:1])
        # losses.append(losses[0])
        output = np.concatenate(outputs)
        losses = np.array(losses)
        decoded_images = np.concatenate(decoded_images)
        # print output
        # print losses
        # np.savetxt(prefix + '_' + params['encoded_save_suffix'], output)

    train_indices = np.mod(np.arange(0, output.shape[0]), output.shape[0] / params['num_angles']) == 0
    train_data = output[train_indices]
    extra_data = output[np.logical_not(train_indices)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(extra_data[:, 1], extra_data[:, 2], extra_data[:, 3], c='blue', marker='.')
    ax.scatter(train_data[:, 1], train_data[:, 2], train_data[:, 3], c='red')
    ax.plot(train_data[:, 1], train_data[:, 2], train_data[:, 3], c='gray')
    fig.show()
    # plt.show()

    fig = plt.figure()
    ax_main = fig.add_subplot(211)
    coords = [0]
    last = 0
    for i in range(1, output.shape[0]):
        last += np.linalg.norm(output[i] - output[i - 1])
        coords.append(last)
    coords = np.array(coords)

    ax_main.plot(coords[np.logical_not(train_indices)], losses[np.logical_not(train_indices)], '.', c='blue')
    ax_main.plot(coords[train_indices], losses[train_indices], c='gray')
    ax_main.plot(coords[train_indices], losses[train_indices], '.', c='red')
    picked, = ax_main.plot([], [], 'o', c='green')

    # fig = plt.figure()
    # plt.axis('off')
    # input_image = np.concatenate(images[:100], axis=1) + 0.5
    # plt.imshow(input_image)
    # print input_image.shape

    ax_input = fig.add_subplot(223)
    ax_input.set_title('Input')
    ax_input.axis('off')

    ax_decoded = fig.add_subplot(224)
    ax_decoded.set_title('Decoded')
    ax_decoded.axis('off')

    best_i = [0]

    def mouse_moved(event):
        changed = False
        if not event.xdata:
            return
        for i, c in enumerate(coords):
            if abs(c - event.xdata) < abs(coords[best_i[0]] - event.xdata):
                best_i[0] = i
                changed = True
        if changed:
            ax_input.imshow(fix_im(images[best_i[0]]))
            ax_input.set_title('Input %d' % best_i[0])
            ax_decoded.imshow(fix_im(decoded_images[best_i[0]]))
            ax_decoded.set_title('Decoded %d' % best_i[0])
            picked.set_data([coords[best_i[0]]], [losses[best_i[0]]])
            fig.canvas.draw()

    fig.canvas.mpl_connect('motion_notify_event', mouse_moved)

    fig.show()

    plt.show()


def main():
    param_file = sys.argv[1]
    with open(param_file) as data_file:
        params = json.load(data_file)

    index = 3
    angles = 64 * 16

    train_images, train_labels = read_and_make_cifar_cycle(params, index, angles)

    handle_cycle(params, train_images, train_labels)


if __name__ == '__main__':
    main()
