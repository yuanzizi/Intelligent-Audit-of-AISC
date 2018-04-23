""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D)
into a point in 2D.
By Youngwook Paul Kwon (young at berkeley.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import system things
from tensorflow.examples.tutorials.mnist import input_data  # for data
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# import helpers
import inference_test
import visualize
import Data_Loader_siamese
import os

# prepare data and tf.session
path = os.path.abspath(os.curdir)

sess = tf.InteractiveSession()

# setup siamese network
siamese = inference_test.siamese()

saver = tf.train.Saver()
tf.initialize_all_variables().run()

# if you just want to load a previously trainmodel?
new = False
model_ckpt = path + '/model3/checkpoint'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = True

# start training
if new:
    m = np.array([[0, 0], [0, 0]])
    saver.restore(sess, path + '/model_1:1_4.11/model.ckpt')
    for k in os.listdir(path + '/1'):
        path_file = os.path.join(path + '/1', k)
        if os.path.isfile(path_file):
            os.remove(path_file)
    for k in os.listdir(path + '/2'):
        path_file = os.path.join(path + '/2', k)
        if os.path.isfile(path_file):
            os.remove(path_file)
    for k in os.listdir(path + '/3'):
        path_file = os.path.join(path + '/3', k)
        if os.path.isfile(path_file):
            os.remove(path_file)
    for k in os.listdir(path + '/4'):
        path_file = os.path.join(path + '/4', k)
        if os.path.isfile(path_file):
            os.remove(path_file)
    for step in range(46):
        batch_x1, batch_x2, batch_y = Data_Loader_siamese.get_test_img('/data1/data_greebear/data/siamese', step)
        loss_v, eucd_v = sess.run([siamese.loss, siamese.eucd], feed_dict={
            siamese.x1: batch_x1,
            siamese.x2: batch_x2,
            siamese.y_: batch_y})
        for i in range(2):
            arr1 = batch_x1[i, :]
            arr2 = batch_x2[i, :]
            arr1 = np.reshape(arr1, [64, 192])
            arr2 = np.reshape(arr2, [64, 192])
            img_list = []
            img_list.extend(arr1)
            img_list.extend(arr2)
            img_arr = np.array(img_list)
            img_arr = np.int8(img_arr * 255)
            img = Image.fromarray(img_arr, 'L')
            acc = (20 - min(20, eucd_v[i]))/0.20
            if batch_y[i] == '1' and eucd_v[i] < 8:
                m[0, 0] = m[0, 0] + 1
                img.save(path + '/1/img%d_%4f_%d%%_%d.jpg' % (step, eucd_v[i], acc, i))
            elif batch_y[i] == '0' and eucd_v[i] < 8:
                m[0, 1] = m[0, 1] + 1
                img.save(path + '/2/img%d_%4f_%d%%_%d.jpg' % (step, eucd_v[i], acc, i))
            elif batch_y[i] == '1' and eucd_v[i] > 8:
                m[1, 0] = m[1, 0] + 1
                img.save(path + '/3/img%d_%4f_%d%%_%d.jpg' % (step, eucd_v[i], acc, i))
            elif batch_y[i] == '0' and eucd_v[i] > 8:
                m[1, 1] = m[1, 1] + 1
                img.save(path + '/4/img%d_%4f_%d%%_%d.jpg' % (step, eucd_v[i], acc, i))
    accuracy = (m[0, 0] + m[1, 1]) / 92.0
    print(m)
    print('accuracy:%g' % (accuracy))
    # print(batch_y)
    # print(eucd_v)

    # save eucd_v
    # print(type(eucd_v))
    # batch_y = list(batch_y)
    # eucd_v = list(eucd_v)
    # file1 = open(path + '/train_index.txt', 'w')
    # for i in range(50):
    #     test_list = [batch_y[i]+' '+eucd_v[i]+'\n']
    # file1.writelines(test_list)
    # file1.close()



else:
    print('Goodbye My Friend .')

