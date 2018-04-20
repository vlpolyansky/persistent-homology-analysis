import tensorflow as tf
import tensorflow.contrib.layers as tflayers
from utils import *
import os
from sklearn.decomposition import IncrementalPCA
import data
import itertools


def train_encode_nn(Encoder,
                    train_features_raw, train_labels_raw,
                    test_features_raw, test_labels_raw,
                    params, **kwargs):
    print 'Data size: ', train_features_raw.shape
    # Directories
    logs_dir = params['logs_dir']
    save_dir = params['save_dir']
    encoding_dir = params['encoding_dir']
    create_dir(logs_dir)
    create_dir('encoded')
    save_file_path = os.path.join(params['save_dir'], 'model.ckpt')

    batch_size = params['batch_size']
    input_shape = params['input_shape']

    # Choice of input style for training
    train_features, train_labels = data.shuffle_data(train_features_raw, train_labels_raw)
    if params['filter_labels'] is not None:
        for i, l in enumerate(params['filter_labels']):
            train_labels[train_labels == l] = i
    max_memory_gb = 1 if 'max_memory_gb' not in params else params['max_memory_gb']
    if train_features.size * 4 < max_memory_gb * 1000000000:  # 1Gb
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(train_features, dtype=tf.float32),
                                                      tf.constant(train_labels, dtype=tf.int64)))
    else:
        dataset = tf.data.Dataset.from_generator(lambda: itertools.izip(train_features, train_labels),
                                                 (tf.float32, tf.int64),
                                                 (tf.TensorShape(input_shape), tf.TensorShape([])))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    # NN for training
    if 'reg_scale' in params:
        encoder = Encoder(trainable=True, reg_scale=params['reg_scale'], name='train')
    else:
        encoder = Encoder(trainable=True, name='train')
    encoder.build_net(features, **kwargs)
    loss, _ = encoder.build_loss(features, labels)

    sess = tf.get_default_session()

    # Training optimizer
    step = tf.Variable(initial_value=0, trainable=False, name='step')
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(params['init_rate'])
        grads = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads, global_step=step)

    merged_summaries = tf.summary.merge_all()

    # NN copy for encoding
    input_ph = tf.placeholder(tf.float32, [None] + input_shape)
    label_ph = tf.placeholder(tf.int64, [None])
    if 'reg_scale' in params:
        encoder2 = Encoder(trainable=False, reg_scale=params['reg_scale'], name='test')
    else:
        encoder2 = Encoder(trainable=False, name='test')
    _, encoded = encoder2.build_net(input_ph, reuse=True, **kwargs)
    loss2, test_summary = encoder2.build_loss(input_ph, label_ph)
    encoding_period = params['encoding_period']
    saving_period = params['saving_period']

    # Encoding helper function
    def encode(data, labels):
        labels_fixed = np.copy(labels)
        if params['filter_labels'] is not None:
            for i, l in enumerate(params['filter_labels']):
                train_labels[train_labels == l] = i
        feed = {
            input_ph: data,
            label_ph: labels_fixed
        }
        encoded_val, test_sum_val = sess.run([encoded, test_summary], feed)
        return np.hstack((labels[:, None], encoded_val)), test_sum_val

    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    writer = tf.summary.FileWriter(logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())

    # Load existing model
    if params['try_load'] and os.path.exists(save_dir):
        try:
            saver.restore(sess, save_file_path)
            my_print('Model restored...\n')
        except:
            my_print('Error restoring model\n')
    create_dir(save_dir)

    my_print('Starting training...\n')
    for i in range(step.eval(), params['train_iter']):
        if i % encoding_period == 0:
            result = sess.run([train_op, step, loss, merged_summaries])
            step_val = result[1]
            loss_val = result[2]
            summary_val = result[3]
            writer.add_summary(summary_val, step_val)
        else:
            result = sess.run([train_op, step, loss])
            step_val = result[1]
            loss_val = result[2]

        if i % encoding_period == 0:
            if params['to_encode']:
                my_print('i = %d: encoding...\n' % i)
                outputs = []
                for j in range(0, train_features_raw.shape[0], batch_size):
                    outputs.append(encode(train_features_raw[j: j + batch_size],
                                          train_labels_raw[j: j + batch_size])[0])
                output = np.concatenate(outputs)
                np.savetxt(os.path.join(encoding_dir, 'train-%d.txt' % i), output)

            if test_features_raw is not None:
                test_choice = np.random.choice(test_features_raw.shape[0], batch_size, False)
                test_accuracy = encode(test_features_raw[test_choice], test_labels_raw[test_choice])[1]
                writer.add_summary(test_accuracy, step_val)

        # if i % 50 == 0:
        #     sys.stderr.write(str(i) + '\n')
        if step_val % (encoding_period / 10) == 0:
            my_print('step: {}, loss: {}\n'.format(step_val, loss_val))

        if step_val % saving_period == 0:
            my_print('Saving model... ')
            saver.save(sess, save_file_path)
            my_print('done\n')


def train_nn(Encoder, train_features, train_labels, params, **kwargs):
    train_features, train_labels = data.shuffle_data(train_features, train_labels)

    logs_dir = params['logs_dir']
    save_dir = params['save_dir']
    create_dir(logs_dir)
    save_file_path = os.path.join(params['save_dir'], 'model.ckpt')

    batch_size = params['batch_size']

    if train_features.size * 4 < 1000000000:  # 1Gb
        dataset = tf.data.Dataset.from_tensor_slices((train_features, tf.constant(train_labels, dtype=tf.int64)))
    else:
        dataset = tf.data.Dataset.from_generator(lambda: itertools.izip(train_features, train_labels),
                                                 (tf.float32, tf.int64), (tf.TensorShape([28, 28]), tf.TensorShape([])))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    encoder = Encoder(trainable=True)
    encoder.build_net(features, **kwargs)

    loss = encoder.build_loss(features, labels)

    step = tf.Variable(initial_value=0, trainable=False, name='step')
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(params['init_rate'])
        grads = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads, global_step=step)

    merged_summaries = tf.summary.merge_all()

    sess = tf.get_default_session()
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    writer = tf.summary.FileWriter(logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())

    if params['try_load'] and os.path.exists(save_dir):
        try:
            saver.restore(sess, save_file_path)
            my_print('Model restored...\n')
        except:
            my_print('Error restoring model\n')
    create_dir(save_dir)

    my_print('Starting training...\n')
    for i in range(step.eval(), params['train_iter']):
        if i % 5000 == 0:
            result = sess.run([train_op, step, loss, merged_summaries])
            step_val = result[1]
            loss_val = result[2]
            summary_val = result[3]
            writer.add_summary(summary_val, step_val)
        else:
            result = sess.run([train_op, step, loss])
            step_val = result[1]
            loss_val = result[2]

        # if i % 50 == 0:
        #     sys.stderr.write(str(i) + '\n')
        if step_val % 500 == 0:
            my_print('step: {}, loss: {}\n'.format(step_val, loss_val))

        if step_val % 20000 == 0:
            my_print('Saving model... ')
            saver.save(sess, save_file_path)
            my_print('done\n')


def encoder_wrapper(Encoder, params, **kwargs):
    save_dir = params['save_dir']
    save_file_path = os.path.join(save_dir, 'model.ckpt')

    input_shape = params['input_shape']
    input_ph = tf.placeholder(tf.float32, [None] + input_shape)

    encoder = Encoder(trainable=True)
    _, encoded = encoder.build_net(input_ph, **kwargs)

    sess = tf.get_default_session()
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    sess.run(tf.global_variables_initializer())

    saver.restore(sess, save_file_path)
    my_print('Model restored...\n')

    def encode(data, labels):
        feed = {
            input_ph: data
        }
        encoded_val = sess.run(encoded, feed)
        return np.hstack((labels[:, None], encoded_val))

    return encode


def encoder_with_decoder_wrapper(Encoder, params, **kwargs):
    save_dir = params['save_dir']
    save_file_path = os.path.join(save_dir, 'model.ckpt')

    input_shape = params['input_shape']
    input_ph = tf.placeholder(tf.float32, [None] + input_shape)

    encoder = Encoder(trainable=True)
    decoded, encoded = encoder.build_net(input_ph, **kwargs)

    sess = tf.get_default_session()
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    sess.run(tf.global_variables_initializer())

    saver.restore(sess, save_file_path)
    my_print('Model restored...\n')

    def encode(data, labels):
        feed = {
            input_ph: data
        }
        encoded_val, decoded_val = sess.run([encoded, decoded], feed)
        return np.hstack((labels[:, None], encoded_val)), decoded_val

    return encode


def decoder_wrapper(Encoder, params, **kwargs):
    save_dir = params['save_dir']
    save_file_path = os.path.join(save_dir, 'model.ckpt')

    input_shape = params['input_shape']
    input_ph = tf.placeholder(tf.float32, [None] + input_shape)

    encoder = Encoder(trainable=True)
    decoded, encoded = encoder.build_net(input_ph, **kwargs)

    sess = tf.Session()
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    sess.run(tf.global_variables_initializer())

    saver.restore(sess, save_file_path)
    my_print('Model restored...\n')

    def decode(data):
        feed = {
            encoded: data
        }
        decoded_val = sess.run(decoded, feed)
        return decoded_val

    return decode


# ---------- PCA --------- #
def calc_pca(data, size):
    pca = IncrementalPCA(size)
    pca.fit(data)
    return pca.transform(data)


# -------- RANDOM -------- #
def process(images, commands):
    for line in commands:
        if line[0] == 'pca':
            images = calc_pca(images, line[1])
        elif line[0] == 'rand_proj':
            images = random_projection(images, line[1])
        else:
            raise Exception('Unknown command: ' + str(line))

    return images


def random_projection(images, dims):
    return images  # TODO
