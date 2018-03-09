import tensorflow as tf


class ConvAutoencoder:
    def __init__(self, trainable):
        self.trainable = trainable
        self.t_encoded = None
        self.t_input = None
        self.t_output = None
        self.t_loss = None

    def build_net(self, t_input, reuse=False, **kwargs):
        self.t_input = t_input

        kernel_sizes = kwargs['kernel_sizes']
        filters_nums = kwargs['filters_nums']
        pool_sizes = kwargs['pool_sizes']
        enc_dim = kwargs['enc_dim']
        input_shape = t_input.shape

        with tf.variable_scope('conv_autoencoder') as scope:
            if reuse:
                scope.reuse_variables()
            net = t_input
            if len(net.shape) == 3:
                net = tf.expand_dims(net, -1)
            sizes = []
            for i, (k_sz, f, p_sz) in enumerate(zip(kernel_sizes, filters_nums, pool_sizes)):
                sizes.append((net.shape[1], net.shape[2]))
                net = tf.layers.conv2d(net, f, (k_sz, k_sz), activation=tf.nn.relu,
                                       padding='same', trainable=self.trainable, name='conv_' + str(i))
                net = tf.layers.max_pooling2d(net, (p_sz, p_sz), (2, 2),
                                              padding='same', name='pool_' + str(i))

            last_shape = net.shape
            net = tf.layers.flatten(net)
            last_len = net.shape[1]
            net = tf.layers.dense(net, enc_dim, activation=tf.nn.tanh,
                                  trainable=self.trainable, name='fc_layer_enc')

            self.t_encoded = net

            net = tf.layers.dense(net, last_len, activation=tf.nn.relu,
                                  trainable=self.trainable, name='fc_layer_dec')
            net = tf.reshape(net, [-1, last_shape[1], last_shape[2], last_shape[3]])

            for i, (k_sz, f, sz) in reversed(list(enumerate(zip(kernel_sizes, filters_nums, sizes)))):
                net = tf.image.resize_images(net, size=sz, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                net = tf.layers.conv2d(inputs=net, filters=f, kernel_size=(k_sz, k_sz), padding='same',
                                       activation=tf.nn.relu)

            net = tf.layers.conv2d(inputs=net, filters=1, kernel_size=(3, 3), padding='same', activation=None)

        net = tf.squeeze(net, [3])

        self.t_output = net

        tf.summary.image('input', tf.expand_dims(t_input, -1))
        tf.summary.image('output', tf.expand_dims(self.t_output, -1))

        return self.t_output, self.t_encoded

    def build_loss(self, t_images, t_labels, loss_name='loss', acc_name='accuracy'):
        self.t_loss = tf.nn.l2_loss(self.t_output - t_images) / tf.cast(tf.shape(t_images)[0], dtype=tf.float32)
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=features + 0.5, logits=net + 0.5))
        loss_summary = tf.summary.scalar(loss_name, self.t_loss)
        return self.t_loss, loss_summary, None


class FCAutoencoder:
    def __init__(self, trainable, reg_scale=0.1, name='autoencoder'):
        self.trainable = trainable
        self.reg_scale = reg_scale
        self.name = name
        self.t_encoded = None
        self.t_input = None
        self.t_output = None
        self.t_loss = None

    def build_net(self, t_input, reuse=False, **kwargs):
        self.t_input = t_input

        layer_sizes = kwargs['fc_layer_sizes']

        with tf.variable_scope('autoencoder') as scope:
            reg = tf.contrib.layers.l2_regularizer(scale=self.reg_scale)
            if reuse:
                scope.reuse_variables()
            net = t_input
            # ASSUMING NET.SHAPE = [N_BATCH, D]

            for i, sz in enumerate(layer_sizes):
                print net.shape
                net = tf.layers.dense(net, sz, activation=tf.nn.tanh,
                                      trainable=self.trainable, name='fc_%d' % i,
                                      kernel_regularizer=reg)
                # if self.trainable:
                #     net = tf.nn.dropout(net, 0.5, name='dropout_' + str(i))

            self.t_encoded = net
            print net.shape

            for i, sz in reversed(list(enumerate([t_input.shape[-1]] + layer_sizes[:-1]))):
                net = tf.layers.dense(net, sz, tf.nn.tanh if i > 0 else None,
                                      trainable=self.trainable, name='fc_back_%d' % i,
                                      kernel_regularizer=reg)
                print net.shape

        self.t_output = net
        return self.t_output, self.t_encoded

    def build_loss(self, t_images, t_labels):
        main_loss = tf.nn.l2_loss(self.t_output - t_images) / tf.cast(tf.shape(t_images)[0], dtype=tf.float32)
        # main_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=t_images + 0.5, logits=self.t_output + 0.5))
        main_loss = tf.identity(main_loss, name='main_loss')
        reg_loss = tf.losses.get_regularization_loss()
        self.t_loss = main_loss + reg_loss
        loss_summary = tf.summary.merge([
            tf.summary.scalar('%s/loss/main' % self.name, main_loss),
            tf.summary.scalar('%s/loss/reg' % self.name, reg_loss),
            tf.summary.scalar('%s/loss' % self.name, self.t_loss),
        ])
        return self.t_loss, loss_summary


class FCClassifier:
    def __init__(self, trainable):
        self.trainable = trainable
        self.t_encoded = None
        self.t_input = None
        self.t_output = None
        self.t_loss = None

    def build_net(self, t_input, reuse=False, **kwargs):
        self.t_input = t_input

        layer_sizes = kwargs['layer_sizes']
        layer_idx = kwargs['layer_idx']
        classes_cnt = kwargs['classes_cnt']

        with tf.variable_scope('classifier') as scope:
            if reuse:
                scope.reuse_variables()
            net = t_input
            net = tf.layers.flatten(net)
            for i, sz in enumerate(layer_sizes):
                net = tf.layers.dense(net, sz, activation=tf.nn.tanh,
                                      trainable=self.trainable, name='layer' + str(i))
                if i == layer_idx:
                    self.t_encoded = net
            net = tf.layers.dense(net, classes_cnt, activation=None,
                                  trainable=self.trainable, name='layer' + str(len(layer_sizes)))

        self.t_output = net
        return self.t_output, self.t_encoded

    def build_loss(self, t_images, t_labels, loss_name='loss', acc_name='accuracy'):
        self.t_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t_labels, logits=self.t_output), name='loss')
        loss_summary = tf.summary.scalar(loss_name, self.t_loss)

        correct_prediction = tf.equal(tf.argmax(self.t_output, 1), t_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.summary.scalar(acc_name, accuracy)

        return self.t_loss, loss_summary, accuracy_summary


class MixedClassifier:
    def __init__(self, trainable, reg=0.0):
        self.trainable = trainable
        self.t_encoded = None
        self.t_input = None
        self.t_output = None
        self.t_loss = None

    def build_net(self, t_input, reuse=False, **kwargs):
        self.t_input = t_input

        kernel_sizes = kwargs['kernel_sizes']
        filters_nums = kwargs['filters_nums']
        pool_sizes = kwargs['pool_sizes']

        layer_sizes = kwargs['fc_layer_sizes']
        layer_idx = kwargs['layer_idx']
        classes_cnt = kwargs['classes_cnt']

        with tf.variable_scope('classifier') as scope:
            if reuse:
                scope.reuse_variables()
            net = t_input
            if len(net.shape) == 3:
                net = tf.expand_dims(net, -1)

            for i, (k_sz, f, p_sz) in enumerate(zip(kernel_sizes, filters_nums, pool_sizes)):
                net = tf.layers.conv2d(net, f, (k_sz, k_sz), activation=tf.nn.relu,
                                       padding='same', trainable=self.trainable, name='conv_' + str(i))
                net = tf.nn.lrn(net, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_%d' % i)
                if i == layer_idx:
                    self.t_encoded = net
                net = tf.layers.max_pooling2d(net, pool_size=(p_sz, p_sz), strides=2,
                                              padding='same', name='pool_' + str(i))

            net = tf.layers.flatten(net)
            add = len(kernel_sizes)
            for i, sz in enumerate(layer_sizes):
                net = tf.layers.dense(net, sz, activation=tf.nn.tanh if i + add == layer_idx else tf.nn.relu,
                                      trainable=self.trainable, name='fc_' + str(i))
                if self.trainable:
                    net = tf.nn.dropout(net, 0.5, name='dropout_' + str(i))
                if i + add == layer_idx:
                    self.t_encoded = net
            net = tf.layers.dense(net, classes_cnt, activation=None,
                                  trainable=self.trainable, name='fc_' + str(len(layer_sizes)))

        self.t_output = net
        return self.t_output, self.t_encoded

    def build_loss(self, t_images, t_labels, loss_name='loss', acc_name='accuracy'):
        self.t_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t_labels, logits=self.t_output), name='loss')
        loss_summary = tf.summary.scalar(loss_name, self.t_loss)

        correct_prediction = tf.equal(tf.argmax(self.t_output, 1), t_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.summary.scalar(acc_name, accuracy)

        return self.t_loss, loss_summary, accuracy_summary


class MixedAutoencoder:
    def __init__(self, trainable, reg_scale=0.1, name='autoencoder'):
        self.trainable = trainable
        self.reg_scale = reg_scale
        self.name = name
        self.t_encoded = None
        self.t_input = None
        self.t_output = None
        self.t_loss = None

    def build_net(self, t_input, reuse=False, **kwargs):
        self.t_input = t_input

        kernel_sizes = kwargs['kernel_sizes']
        filters_nums = kwargs['filters_nums']
        pool_sizes = kwargs['pool_sizes']
        layer_sizes = kwargs['fc_layer_sizes']

        image_sizes = []

        with tf.variable_scope('autoencoder') as scope:
            reg = tf.contrib.layers.l2_regularizer(scale=self.reg_scale)
            if reuse:
                scope.reuse_variables()
            net = t_input
            if len(net.shape) == 3:
                net = tf.expand_dims(net, -1)

            for i, (k_sz, f_num, p_sz) in enumerate(zip(kernel_sizes, filters_nums, pool_sizes)):
                print net.shape
                image_sizes.append(net.shape[1])
                net = tf.layers.conv2d(net, f_num, (k_sz, k_sz), activation=tf.nn.relu,
                                       padding='same', trainable=self.trainable, name='conv_%d' % i,
                                       kernel_regularizer=reg)
                net = tf.nn.lrn(net, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_%d' % i)
                net = tf.layers.max_pooling2d(net, pool_size=(p_sz, p_sz), strides=2,
                                              padding='same', name='pool_%d' % i)

            last_shape = net.shape
            net = tf.layers.flatten(net)

            for i, sz in enumerate(layer_sizes):
                print net.shape
                net = tf.layers.dense(net, sz, activation=tf.nn.tanh if i + 1 == len(layer_sizes) else tf.nn.relu,
                                      trainable=self.trainable, name='fc_%d' % i,
                                      kernel_regularizer=reg)
                # if self.trainable:
                #     net = tf.nn.dropout(net, 0.5, name='dropout_' + str(i))

            # self.t_encoded = tf.identity(net, 'encoded')
            self.t_encoded = net

            for i, sz in reversed(list(enumerate([last_shape[1] * last_shape[2] * last_shape[3]] + layer_sizes[:-1]))):
                net = tf.layers.dense(net, sz, tf.nn.relu,
                                      trainable=self.trainable, name='fc_back_%d' % i,
                                      kernel_regularizer=reg)
                print net.shape

            net = tf.reshape(net, [-1, last_shape[1], last_shape[2], last_shape[3]])

            for i, (k_sz, f_num, im_sz) in reversed(list(enumerate(zip(kernel_sizes,
                                                                       [t_input.shape[-1]] + filters_nums[:-1],
                                                                       image_sizes)))):
                net = tf.image.resize_images(net, size=(im_sz, im_sz), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                net = tf.layers.conv2d(inputs=net, filters=f_num, kernel_size=(k_sz, k_sz), padding='same',
                                       activation=tf.nn.relu if i > 0 else None, name='deconv_%d' % i,
                                       kernel_regularizer=reg)
                if i > 0:
                    net = tf.nn.lrn(net, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_%d' % i)
                print net.shape

        self.t_output = net
        return self.t_output, self.t_encoded

    def build_loss(self, t_images, t_labels):
        main_loss = tf.nn.l2_loss(self.t_output - t_images) / tf.cast(tf.shape(t_images)[0], dtype=tf.float32)
        # main_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=t_images + 0.5, logits=self.t_output + 0.5))
        main_loss = tf.identity(main_loss, name='main_loss')
        reg_loss = tf.losses.get_regularization_loss()
        self.t_loss = main_loss + reg_loss
        loss_summary = tf.summary.merge([
            tf.summary.scalar('%s/loss/main' % self.name, main_loss),
            tf.summary.scalar('%s/loss/reg' % self.name, reg_loss),
            tf.summary.scalar('%s/loss' % self.name, self.t_loss),
            tf.summary.image('%s/input' % self.name, self.t_input),
            tf.summary.image('%s/output' % self.name, self.t_output)
        ])
        return self.t_loss, loss_summary


class SphericalEncoder:
    def __init__(self, trainable, reg_scale=0.1, sph_scale=1.0, name='autoencoder'):
        self.trainable = trainable
        self.reg_scale = reg_scale
        self.sph_scale = sph_scale
        self.name = name
        self.t_encoded = None
        self.t_input = None
        self.t_output = None
        self.t_loss = None

    def build_net(self, t_input, reuse=False, **kwargs):
        self.t_input = t_input

        layer_sizes = kwargs['fc_layer_sizes']

        with tf.variable_scope('autoencoder') as scope:
            reg = tf.contrib.layers.l2_regularizer(scale=self.reg_scale)
            if reuse:
                scope.reuse_variables()
            net = t_input
            # ASSUMING NET.SHAPE = [N_BATCH, D]

            for i, sz in enumerate(layer_sizes):
                print net.shape
                net = tf.layers.dense(net, sz, activation=tf.nn.tanh,
                                      trainable=self.trainable, name='fc_%d' % i,
                                      kernel_regularizer=reg)
                # if self.trainable:
                #     net = tf.nn.dropout(net, 0.5, name='dropout_' + str(i))

            self.t_encoded = net
            print net.shape

            for i, sz in reversed(list(enumerate([t_input.shape[-1]] + layer_sizes[:-1]))):
                net = tf.layers.dense(net, sz, tf.nn.tanh if i > 0 else None,
                                      trainable=self.trainable, name='fc_back_%d' % i,
                                      kernel_regularizer=reg)
                print net.shape

        self.t_output = net
        return self.t_output, self.t_encoded

    def build_loss(self, t_images, t_labels):
        main_loss = tf.nn.l2_loss(self.t_output - t_images) / tf.cast(tf.shape(t_images)[0], dtype=tf.float32)
        # main_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=t_images + 0.5, logits=self.t_output + 0.5))
        main_loss = tf.identity(main_loss, name='main_loss')
        reg_loss = tf.losses.get_regularization_loss()
        sph_loss = tf.abs(tf.nn.l2_loss(self.t_encoded) - 0.75) * self.sph_scale
        self.t_loss = main_loss + reg_loss + sph_loss
        loss_summary = tf.summary.merge([
            tf.summary.scalar('%s/loss/main' % self.name, main_loss),
            tf.summary.scalar('%s/loss/reg' % self.name, reg_loss),
            tf.summary.scalar('%s/loss/sph' % self.name, sph_loss),
            tf.summary.scalar('%s/loss' % self.name, self.t_loss),
        ])
        return self.t_loss, loss_summary
