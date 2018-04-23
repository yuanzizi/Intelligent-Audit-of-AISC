import tensorflow as tf
from tensorflow.contrib import slim


class siamese:

    # Create model
    def __init__(self):
        self.regularization_rate = 0.0005
        self.x1 = tf.placeholder(tf.float32, [None, 64, 192, 1])
        self.x2 = tf.placeholder(tf.float32, [None, 64, 192, 1])

        # with tf.variable_scope("siamese") as scope:
        self.o = self.network(self.x1, self.x2)
        # scope.reuse_variables()
        # self.o2 = self.network(x=self.x2)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss, self.ob = self.loss_with_spring()
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def network(self, x1, x2, training_flag=True):
        x = tf.concat([x1, x2], axis=3)
        with tf.variable_scope("net") as scope1:
            # 64*192
            w1 = tf.get_variable("w1", shape=[3, 3, 2, 64],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            conv1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
            conv1 = tf.layers.batch_normalization(conv1, training=training_flag)
            conv1 = tf.nn.relu(conv1)

            w2 = tf.get_variable("w2", shape=[3, 3, 64, 64],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            conv2 = tf.nn.conv2d(conv1, w2, strides=[1, 1, 1, 1], padding='SAME')
            conv2 = tf.layers.batch_normalization(conv2, training=training_flag)
            conv2 = tf.nn.relu(conv2)

            pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # 32*96
            w3 = tf.get_variable("w3", shape=[3, 3, 64, 128],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            conv3 = tf.nn.conv2d(pool1, w3, strides=[1, 1, 1, 1], padding='SAME')
            conv3 = tf.layers.batch_normalization(conv3, training=training_flag)
            conv3 = tf.nn.relu(conv3)

            w8 = tf.get_variable("w8", shape=[3, 3, 128, 128],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            conv8 = tf.nn.conv2d(conv3, w8, strides=[1, 1, 1, 1], padding='SAME')
            conv8 = tf.layers.batch_normalization(conv8, training=training_flag)
            conv8 = tf.nn.relu(conv8)

            pool2 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # 16*48
            w4 = tf.get_variable("w4", shape=[3, 3, 128, 256],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            conv4 = tf.nn.conv2d(pool2, w4, strides=[1, 1, 1, 1], padding='SAME')
            conv4 = tf.layers.batch_normalization(conv4, training=training_flag)
            conv4 = tf.nn.relu(conv4)

            pool3 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # 8*24
            w5 = tf.get_variable("w5", shape=[3, 3, 256, 512],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            conv5 = tf.nn.conv2d(pool3, w5, strides=[1, 1, 1, 1], padding='SAME')
            conv5 = tf.layers.batch_normalization(conv5, training=training_flag)
            conv5 = tf.nn.relu(conv5)

            # flatten
            flatten = slim.flatten(conv5)

            # fc1
            w6 = tf.get_variable("w6", shape=[8 * 24 * 512, 1024],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            b6 = tf.get_variable("b6", shape=[1024],
                                 dtype=tf.float32, initializer=tf.zeros_initializer())
            fc1 = tf.matmul(flatten, w6) + b6
            fc1 = tf.layers.batch_normalization(fc1, training=training_flag)
            fc1 = tf.nn.relu(fc1)

            # dropout
            # conv6 = slim.dropout(conv6, 0.8, is_training=training_flag)

            # fc2
            w7 = tf.get_variable("w7", shape=[1024, 1],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            b7 = tf.get_variable("b7", shape=[1],
                                 dtype=tf.float32, initializer=tf.zeros_initializer())
            pred = tf.matmul(fc1, w7) + b7
            #pred = tf.nn.sigmoid(pred)

            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w1))
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w2))
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w3))
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w4))
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w5))
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w6))
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w7))
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w8))

        return pred


    def loss_with_spring(self):
        label = self.y_
        label = tf.expand_dims(label, -1)
        error_loss = tf.reduce_sum(tf.maximum(0.0, 1.0-tf.multiply(label, self.o)))
        regularization_loss = tf.add_n(tf.get_collection('losses'))
        loss = regularization_loss + error_loss
        see=tf.maximum(0.0, 1.0-tf.multiply(label, self.o))

        return loss, see
