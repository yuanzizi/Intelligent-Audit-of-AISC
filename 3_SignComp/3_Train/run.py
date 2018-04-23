# --coding:utf-8--


""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D)
into a point in 2D.
By Youngwook Paul Kwon (young at berkeley.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import system things
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import numpy as np
import os

#import helpers
import inference
import visualize
import Data_Loader_siamese

# prepare data and tf.session
mnist = input_data.read_data_sets('/data1/data_greebear/gree/siamese/MNIST_data', one_hot=False)
sess = tf.InteractiveSession()

# setup siamese network
siamese = inference.siamese();

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


'''
var_list = tf.trainable_variables()
restore_varlist = var_list[0:11]
trainable_varlist = var_list[12:16]
saver = tf.train.Saver(var_list=restore_varlist)
'''
saver = tf.train.Saver()

with tf.control_dependencies(update_ops):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)#, var_list=trainable_varlist)

tf.initialize_all_variables().run()
saver.restore(sess, '/data1/data_greebear/gree/siamese/model3/model.ckpt')
# if you just want to load a previously trainmodel?
new = True
model_ckpt = '/data1/data_greebear/gree/siamese/model3/checkpoint'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False

# start training
if new:
    for step in range(100000):
        # batch_x1, batch_y1 = mnist.train.next_batch(128)
        # batch_x2, batch_y2 = mnist.train.next_batch(128)
        # batch_y = (batch_y1 == batch_y2).astype('float')
        batch_x1, batch_x2, batch_y = Data_Loader_siamese.get_train_img('/data1/data_greebear/data/siamese', 32)
        _, loss_v, eucd_v = sess.run([train_step, siamese.loss, siamese.eucd], feed_dict={
                            siamese.x1: batch_x1,
                            siamese.x2: batch_x2,
                            siamese.y_: batch_y})
        
        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()

        if step % 10 == 0:
            print ('step %d: loss %.3f' % (step, loss_v))

        if step % 1000 == 0 and step > 0:
            saver.save(sess, '/data1/data_greebear/gree/siamese/model3/model.ckpt')
            # embed = siamese.o1.eval({siamese.x1: mnist.test.images})
            # embed.tofile('/data1/data_greebear/gree/siamese/embed.txt')
else:
    saver.restore(sess, '/data1/data_greebear/ifeng/lenet_finetuning/model/model.ckpt')

# visualize result
# embed = siamese.o1.eval({siamese.x1: mnist.test.images})
# embed.tofile('./embed.txt')
# x_test = mnist.test.images.reshape([-1, 28, 28])
# visualize.visualize(embed, x_test)
print('haha')