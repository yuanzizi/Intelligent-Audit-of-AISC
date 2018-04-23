# -*- coding: utf-8 -*-

# 检测图片的方向，并转到正方向

import tensorflow as tf
import glob
import os
import io
import numpy as np
from model1_0315 import model
#from  PIL import Image
import shutil
from wand.image import Image as WImage
from wand.display import display
from wand.color import Color
from PIL import Image as PImage

PATH_DATA = os.path.join('..','..','OUTPUT','img4label')
RECORD_NAME = os.path.join(PATH_DATA,'rotate.tfrecord') # 储存转换后的图片，预测后可以删除
PATH_MODEL = os.path.join('..','..','model3')


def write_tfrecord(names, size=200): # 将文件夹下的图片写入tfrecord
    
    bg = Color('white') # 设置要切换的边缘颜色，此处为白色

    with tf.python_io.TFRecordWriter(RECORD_NAME) as writer:
    
        for name in names:
            img = WImage(filename=name)
            img.trim(color=bg, fuzz=20) # 切除白块，20是测试出来的
            img.resize(size, size)
            img_buffer = np.asarray(bytearray(img.make_blob(format='RGB')),dtype=np.uint8)
            img = PImage.fromarray(img_buffer.reshape(200,200,3))
#            img.save(os.path.join(PATH_OUT,'temp.jpg'))
#            img = PImage.open(os.path.join(PATH_OUT,'temp.jpg'))
#            byte_img = img.convert('RGB') # 有些图片是黑白灰度，需要转换成RGB
#            byte_img = byte_img.resize((200,200))
            byte_img = img.tobytes()
            byte_name = bytes(name, encoding='utf-8')
#                print('write:',label_true, byte_name, len(byte_img))
            tf_feature = {'byte_img':tf.train.Feature(bytes_list=tf.train.BytesList(value=[byte_img])),
                          'byte_name':tf.train.Feature(bytes_list=tf.train.BytesList(value=[byte_name]))}   
            tf_features = tf.train.Features(feature=tf_feature)
            example = tf.train.Example(features=tf_features)
            writer.write(example.SerializeToString()) 

def read_decode(serialized_example):

    tf_features = {'byte_img':tf.FixedLenFeature([], tf.string),
                   'byte_name':tf.FixedLenFeature([], tf.string)}
    
    features = tf.parse_single_example(
            serialized_example,
            features=tf_features)
    
    img = tf.decode_raw(features['byte_img'], tf.uint8)
    img = 2*tf.cast(img, tf.float32) * (1./255) -1
    img = tf.reshape(img, [200,200,3])
    name = tf.cast(features['byte_name'],tf.string)
    return img, name

def decode_name(name):
    return str(name,encoding='utf8')       

def read_copy(folder):
    
    PATH_OUT = os.path.join(PATH_DATA, folder+'_rotate')
    if not os.path.exists(os.path.join(PATH_OUT)):
        os.mkdir(os.path.join(PATH_OUT))
    
    dataset = tf.data.TFRecordDataset(RECORD_NAME)
    dataset = dataset.map(read_decode)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(5)
    iterator =  dataset.make_one_shot_iterator()
    
    batch_image, batch_name = iterator.get_next()
#    batch_image = iterator.get_next()

    score_label = model(batch_image, False)
    pred_label = tf.argmax(score_label, 1)
    softmax = tf.nn.softmax(score_label)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, os.path.join(PATH_MODEL,'model3.ckpt')) 

        while True:
            try:
                label_pred,name = sess.run([pred_label, batch_name])

                name = list(map(decode_name, name))
#                print(label_pred, name)
                for p,n in zip(label_pred, name):
                    print(p,n)


#                n = os.path.split(name)
#                    out_n = os.path.join(PATH_OUT,str(p),'%s_%s'%(os.path.split(n)[0][-29:],os.path.split(n)[1]))
#                    shutil.copyfile(n,out_n)
            except:
                print('Finish')
                break
            
#    return y_true, y_pred, names, label_prob,label_score
tf.reset_default_graph() 

folders = os.listdir(PATH_DATA)

for folder in folders:
    tf.reset_default_graph()
    if folder not in ['5','7']:
        continue
    img_paths = glob.glob(os.path.join(PATH_DATA, folder, '*.jpg'))
    write_tfrecord(img_paths)
    read_copy(folder)
    