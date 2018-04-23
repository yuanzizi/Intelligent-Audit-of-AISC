# -- coding: utf-8 --
import tensorflow as tf
from tensorflow.contrib import slim


def model(inputs, is_train=True):
    # 200*200
    d_w1 = tf.get_variable("dw_1", shape=[5, 5, 3, 64],
                           dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer1 = tf.nn.conv2d(inputs, d_w1, strides=[1, 1, 1, 1], padding='SAME')
    layer1 = tf.nn.relu(layer1)

    d_w2 = tf.get_variable("dw_2", shape=[5, 5, 64, 64],
                           dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer2 = tf.nn.conv2d(layer1, d_w2, strides=[1, 1, 1, 1], padding='SAME')
    layer2 = tf.layers.batch_normalization(layer2, training=is_train)

    pool = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 100*100
    d_w3 = tf.get_variable("dw_3", shape=[4, 4, 64, 88],
                           dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer3 = tf.nn.conv2d(pool, d_w3, strides=[1, 1, 1, 1], padding='SAME')
    layer3 = tf.layers.batch_normalization(layer3, training=is_train)
    layer3 = tf.nn.relu(layer3)

    d_w4 = tf.get_variable("d_w4", shape=[4, 4, 88, 88],
                           dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer4 = tf.nn.conv2d(layer3, d_w4, strides=[1, 1, 1, 1], padding='SAME')
    layer4 = tf.layers.batch_normalization(layer4, training=is_train)
    layer4 = tf.nn.relu(layer4)

    pool = tf.nn.max_pool(layer4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 50*50
    d_w5 = tf.get_variable("d_w5", shape=[3, 3, 88, 112],
                           dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer5 = tf.nn.conv2d(pool, d_w5, strides=[1, 1, 1, 1], padding='SAME')
    layer5 = tf.layers.batch_normalization(layer5, training=is_train)
    layer5 = tf.nn.relu(layer5)

    d_w6 = tf.get_variable("d_w6", shape=[3, 3, 112, 112],
                           dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer6 = tf.nn.conv2d(layer5, d_w6, strides=[1, 1, 1, 1], padding='SAME')
    layer6 = tf.layers.batch_normalization(layer6, training=is_train)
    layer6 = tf.nn.relu(layer6)

    pool = tf.nn.max_pool(layer6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 25*25
    d_w7 = tf.get_variable("d_w7", shape=[3, 3, 112, 136],
                           dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer7 = tf.nn.conv2d(pool, d_w7, strides=[1, 1, 1, 1], padding='SAME')
    layer7 = tf.layers.batch_normalization(layer7, training=is_train)
    layer7 = tf.nn.relu(layer7)

    d_w8 = tf.get_variable("d_w8", shape=[3, 3, 136, 136],
                           dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer8 = tf.nn.conv2d(layer7, d_w8, strides=[1, 1, 1, 1], padding='SAME')
    layer8 = tf.layers.batch_normalization(layer8, training=is_train)
    layer8 = tf.nn.relu(layer8)

    pool = tf.nn.max_pool(layer8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 13*13
    d_w9 = tf.get_variable("d_w9", shape=[3, 3, 136, 160],
                           dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer9 = tf.nn.conv2d(pool, d_w9, strides=[1, 1, 1, 1], padding='SAME')
    layer9 = tf.layers.batch_normalization(layer9, training=is_train)
    layer9 = tf.nn.relu(layer9)

    d_w10 = tf.get_variable("d_w10", shape=[3, 3, 160, 160],
                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer10 = tf.nn.conv2d(layer9, d_w10, strides=[1, 1, 1, 1], padding='SAME')
    layer10 = tf.layers.batch_normalization(layer10, training=is_train)
    layer10 = tf.nn.relu(layer10)

    """
    pool = tf.nn.max_pool(layer10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 7*7
    d_w11 = tf.get_variable("d_w11", shape=[3, 3, 128, 144],
                           dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer11 = tf.nn.conv2d(pool, d_w11, strides=[1, 1, 1, 1], padding='SAME')
    layer11 = tf.layers.batch_normalization(layer11, training=is_train)
    layer11 = tf.nn.relu(layer11)

    d_w12 = tf.get_variable("d_w12", shape=[3, 3, 144, 144],
                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer12 = tf.nn.conv2d(layer11, d_w12, strides=[1, 1, 1, 1], padding='SAME')
    layer12 = tf.layers.batch_normalization(layer12, training=is_train)
    layer12 = tf.nn.relu(layer12)
    """
    # global downsampling
    d_w13 = tf.get_variable("d_w13", shape=[13, 13, 160, 360],
                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    layer13 = tf.nn.conv2d(layer10, d_w13, strides=[1, 1, 1, 1], padding='VALID')
    layer13 = tf.layers.batch_normalization(layer13, training=is_train)
    layer13 = tf.nn.relu(layer13)
    layer13 = slim.flatten(layer13)

    layer13 = slim.dropout(layer13, 0.8, is_training=is_train)

    # 分类
    d_w14 = tf.get_variable("d_w14", shape=[360, 8],
                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
    b14 = tf.get_variable("b14", shape=[8],
                          dtype=tf.float32, initializer=tf.zeros_initializer())
    net = tf.matmul(layer13, d_w14) + b14
    return net

