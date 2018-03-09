import tensorflow as tf
import json
import numpy as np
import idx2numpy
import sys
from matplotlib.pyplot import *
# from data import normalize_data
import data
import algo
import encoders
import utils


def extract_kwargs(params):
    kwargs = dict()
    choices = {
        # 'ConvAutoencoder': {
        #     'kernel_sizes': 'kernel_sizes',
        #     'filters_nums': 'filters_nums',
        #     'pool_sizes': 'pool_sizes',
        #     'enc_dim': 'dim'
        # },
        'FCAutoencoder': {
            'fc_layer_sizes': 'fc_layers'
        },
        'SphericalEncoder': {
            'fc_layer_sizes': 'fc_layers'
        },
        # 'FCClassifier': {
        #     'layer_sizes': 'clf_layers',
        #     'layer_idx': 'clf_layer_idx',
        #     'classes_cnt': 'classes'
        # },
        # 'MixedClassifier': {
        #     'kernel_sizes': 'kernel_sizes',
        #     'filters_nums': 'filters_nums',
        #     'pool_sizes': 'pool_sizes',
        #     'fc_layer_sizes': 'fc_layers',
        #     'layer_idx': 'enc_layer_idx',
        #     'classes_cnt': 'classes'
        # },
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


def prepare_data(params):
    if ('data_name' not in params) or params['data_name'] == 'mnist':
        read_data = data.read_mnist_data
    elif params['data_name'] == 'cifar':
        read_data = data.read_cifar_data
    elif params['data_name'] == 'cmu':
        read_data = data.read_unlabeled_data
    elif params['data_name'] == 'vector':
        read_data = data.read_vector_data
    else:
        raise Exception('Unknown data_name: %s' % params['data_name'])

    train_images, train_labels, test_images, test_labels = read_data(params)

    utils.create_dir('_data')
    np.save('_data/train_images.npy', train_images)  # todo replace with params names
    if train_labels is not None:
        np.save('_data/train_labels.npy', train_labels)
    if test_images is not None:
        np.save('_data/test_images.npy', test_images)
    if test_labels is not None:
        np.save('_data/test_labels.npy', test_labels)


def main_train(params):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        train_images = np.load(params['data_folder'] + '/train_images.npy')
        train_labels = np.load(params['data_folder'] + '/train_labels.npy')
        encoder = getattr(encoders, params['algo'])
        algo.train_nn(encoder, train_images, train_labels, params, **extract_kwargs(params))


def main_train_encode(params):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        train_images = np.load(params['data_folder'] + '/train_images.npy')
        train_labels = np.load(params['data_folder'] + '/train_labels.npy')
        try:
            test_images = np.load(params['data_folder'] + '/test_images.npy')
            test_labels = np.load(params['data_folder'] + '/test_labels.npy')
        except IOError as e:
            print 'IOError on import of test data'
            print e
            test_images = None
            test_labels = None
        encoder = getattr(encoders, params['algo'])
        algo.train_encode_nn(encoder, train_images, train_labels, test_images, test_labels,
                             params, **extract_kwargs(params))


def identity_encode(params):
    train_images = np.load(params['data_folder'] + '/train_images.npy')
    train_labels = np.load(params['data_folder'] + '/train_labels.npy')
    np.savetxt('train_' + params['encoded_save_suffix'], np.hstack((train_labels[:, None],
                                                                    train_images.reshape([train_images.shape[0], -1]))))

    test_images = np.load(params['data_folder'] + '/test_images.npy')
    test_labels = np.load(params['data_folder'] + '/test_labels.npy')
    np.savetxt('test_' + params['encoded_save_suffix'], np.hstack((test_labels[:, None],
                                                                    test_images.reshape([test_images.shape[0], -1]))))


def main_encode(params):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        encoder = getattr(encoders, params['algo'])

        encode = algo.encoder_wrapper(encoder, params, **extract_kwargs(params))

        def handle(prefix, images, labels):
            outputs = []
            step = 500
            for i in range(0, images.shape[0], step):
                outputs.append(encode(images[i: i + step], labels[i: i + step]))
            output = np.concatenate(outputs)
            np.savetxt(prefix + '_' + params['encoded_save_suffix'], output)

        train_images = np.load(params['data_folder'] + '/train_images.npy')
        train_labels = np.load(params['data_folder'] + '/train_labels.npy')
        handle('train', train_images, train_labels)

        try:
            test_images = np.load(params['data_folder'] + '/test_images.npy')
            test_labels = np.load(params['data_folder'] + '/test_labels.npy')
            handle('test', test_images, test_labels)
        except IOError as e:
            print 'IOError on test data'
            print e


def process_data(params):

    def handle(some_data):
        labels = some_data[:, 0:1]
        images = some_data[:, 1:]

        images = algo.process(images, params['postprocess'])
        return np.hstack((labels, images))

    train_data = np.loadtxt('train_' + params['encoded_save_suffix'])
    np.savetxt('train_' + params['encoded_save_suffix'], handle(train_data))

    try:
        test_data = np.loadtxt('test_' + params['encoded_save_suffix'])
        np.savetxt('test_' + params['encoded_save_suffix'], handle(test_data))
    except IOError as e:
        print 'IOError on test data'
        print e


def main_pca(params):
    train_images, train_labels, test_images, test_labels = data.read_mnist_data(params)

    def handle(prefix, images, labels):
        pca = algo.calc_pca(np.reshape(images, [-1, 28 * 28]), params['dim'])
        output = np.hstack((labels[:, None], pca))
        np.savetxt(prefix + '_' + params['encoded_save_suffix'], output)

    handle('train', train_images, train_labels)
    handle('test', test_images, test_labels)


def main():
    func_map = {
        'train': main_train,  # obsolete
        'identity': identity_encode,
        'encode': main_encode,
        'train_encode': main_train_encode,
        'pca': main_pca,
        'prepare': prepare_data,
        'process': process_data,
    }

    param_file = 'params.json'
    if len(sys.argv) > 2:
        param_file = sys.argv[2]
    with open(param_file) as data_file:
        params = json.load(data_file)

    func_map[sys.argv[1]](params)


if __name__ == '__main__':
    main()
