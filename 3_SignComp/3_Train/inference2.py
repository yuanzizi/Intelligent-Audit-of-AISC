import tensorflow as tf
from tensorflow.contrib import slim

class siamese:

    # Create model
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 64, 192, 1])
        self.x2 = tf.placeholder(tf.float32, [None, 64, 192, 1])
        self.regularization_rate = 0.0005

        #with tf.variable_scope("siamese") as scope:
        self.o = self.network(self.x1, self.x2)
        #scope.reuse_variables()
        # self.o2 = self.network(x=self.x2)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def network(self, x1, x2, training_flag=True):
        with tf.variable_scope("net") as scope:
            # 64*192
            w1 = tf.get_variable("w1", shape=[3, 3, 1, 64],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            conv1 = tf.nn.conv2d(x1, w1, strides=[1, 1, 1, 1], padding='SAME')
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
            
            w3_2 = tf.get_variable("w3_2", shape=[3, 3, 128, 128],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            conv3_2 = tf.nn.conv2d(conv3, w3_2, strides=[1, 1, 1, 1], padding='SAME')
            conv3_2 = tf.layers.batch_normalization(conv3_2, training=training_flag)
            conv3_2 = tf.nn.relu(conv3_2)
    
            pool2 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
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
    
            #flatten
            flatten = slim.flatten(conv5)
    
            #fc1
            w6 = tf.get_variable("w6", shape=[8*24*512, 1024],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            b6 = tf.get_variable("b6", shape=[1024],
                                 dtype=tf.float32, initializer=tf.zeros_initializer())
            fc1 = tf.matmul(flatten, w6) + b6
            fc1 = tf.layers.batch_normalization(fc1, training=training_flag)
            fc1 = tf.nn.relu(fc1)
    
            # dropout
            #conv6 = slim.dropout(conv6, 0.8, is_training=training_flag)
    
            # fc2
            w7 = tf.get_variable("w7", shape=[1024, 256],
                                   dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            b7 = tf.get_variable("b7", shape=[256],
                                 dtype=tf.float32, initializer=tf.zeros_initializer())
            pred1 = tf.matmul(fc1, w7) + b7
            
            tf.add_to_collection('weights', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w1))
            tf.add_to_collection('weights', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w2))
            tf.add_to_collection('weights', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w3))
            tf.add_to_collection('weights', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w4))
            tf.add_to_collection('weights', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w5))
            tf.add_to_collection('weights', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w6))
            tf.add_to_collection('weights', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w7))
            tf.add_to_collection('weights', tf.contrib.layers.l2_regularizer(self.regularization_rate)(w3_2))

        with tf.variable_scope("net") as scope:
            scope.reuse_variables()
            # 64*192
            w1 = tf.get_variable("w1", shape=[3, 3, 1, 64],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            conv1 = tf.nn.conv2d(x2, w1, strides=[1, 1, 1, 1], padding='SAME')
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
            
            w3_2 = tf.get_variable("w3_2", shape=[3, 3, 128, 128],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            conv3_2 = tf.nn.conv2d(conv3, w3_2, strides=[1, 1, 1, 1], padding='SAME')
            conv3_2 = tf.layers.batch_normalization(conv3_2, training=training_flag)
            conv3_2 = tf.nn.relu(conv3_2)
    
            pool2 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
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
    
            #flatten
            flatten = slim.flatten(conv5)
    
            #fc1
            w6 = tf.get_variable("w6", shape=[8*24*512, 1024],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            b6 = tf.get_variable("b6", shape=[1024],
                                 dtype=tf.float32, initializer=tf.zeros_initializer())
            fc1 = tf.matmul(flatten, w6) + b6
            fc1 = tf.layers.batch_normalization(fc1, training=training_flag)
            fc1 = tf.nn.relu(fc1)
    
            # dropout
            #conv6 = slim.dropout(conv6, 0.8, is_training=training_flag)
    
            # fc2
            w7 = tf.get_variable("w7", shape=[1024, 256],
                                   dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
            b7 = tf.get_variable("b7", shape=[256],
                                 dtype=tf.float32, initializer=tf.zeros_initializer())
            pred2 = tf.matmul(fc1, w7) + b7
            
        

        pred = tf.pow(tf.subtract(pred1, pred2), 2)

        # fc
        # d_w9 = tf.get_variable("d_w9", shape=[256, 512],
        #                        dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
        # b9 = tf.get_variable("b9", shape=[512],
        #                      dtype=tf.float32, initializer=tf.zeros_initializer())
        # pred3 = tf.nn.relu(tf.matmul(feature, d_w9) + b9)
        #
        # # fc
        # d_w10 = tf.get_variable("d_w10", shape=[512, 2],
        #                         dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(False))
        # b10 = tf.get_variable("b10", shape=[2],
        #                       dtype=tf.float32, initializer=tf.zeros_initializer())
        # pred = tf.matmul(pred3, d_w10) + b10

        return pred

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def loss_with_spring(self):
        margin = 10.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        # eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(self.o, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        self.eucd = eucd
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, 2*neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")+tf.add_n(tf.get_collection('weights'))
        return loss