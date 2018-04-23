# -- coding: utf-8 --

import numpy as np
import tensorflow as tf
from PIL import Image
from model3_0315 import model
import os

training_steps = 20000

path = "/home/fly/HLF/model1_linux"
path = os.path.pardir
trainfile = os.path.join(path ,"model3_train.tfrecords")
testfile = os.path.join(path,"model3_test.tfrecords")
tf.reset_default_graph() 

#tensorboard_path = "/home/ly/Desktop/Deform/model1_linux/logs"

# tfrecord解码
def read_and_decode(filename, batch_size, capacity, min_after_dequeue):
    filename_queue = tf.train.string_input_producer([filename],shuffle=False) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image': tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象

    image = tf.decode_raw(features['image'], tf.uint8)
    # 将数据变化[-1,1]
    image = 2*tf.cast(image, tf.float32) * (1. / 255) - 1
    image = tf.reshape(image, [200, 200, 3])
    label = tf.cast(features['label'], tf.int64)
    # 生成batch
    batch_image, batch_label = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                      capacity=capacity, min_after_dequeue=min_after_dequeue)
    return batch_image, batch_label

train_img, train_label = read_and_decode(trainfile, batch_size=16, capacity=32, min_after_dequeue=16)

outputs = model(train_img)

loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=train_label-1))
tf.summary.scalar('loss', loss)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step = tf.train.AdamOptimizer(0.0002).minimize(loss)
correct_prediction = tf.equal(tf.argmax(outputs, 1), train_label-1)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy', accuracy) 

summary = tf.summary.merge_all() 

saver = tf.train.Saver()

#with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))) as sess:
with tf.Session() as sess:
    #summary_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(training_steps):
        sess.run(train_step)

        if i % 100 == 0:
            #summary_writer.add_summary(sess.run(summary), i)
            print("After %d training step(s), loss is %g, acc is %g " % (i, sess.run(loss), sess.run(accuracy)))
        if i% 1000 ==0 and i >0 :
            saver.save(sess, path+"/model/model3.ckpt")
    coord.request_stop()
    coord.join(threads)