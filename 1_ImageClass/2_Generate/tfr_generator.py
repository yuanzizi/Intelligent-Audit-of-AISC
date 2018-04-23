# -- coding: utf-8 --

import os
import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.ndimage
import random



path = "E:/workspace/python/pycharm/ChinaMobile/model1"
path = os.path.join(os.path.pardir,'split')
filename1 = os.path.join(path ,"model1_train.tfrecords")
filename2 = os.path.join(path ,"model1_test.tfrecords")
np.random.seed(2018)

def tfrecords_generator(filename):
    writer = tf.python_io.TFRecordWriter(filename)  # 要生成的文件
    i = 0
    if "train" in filename:
        for class_num in os.listdir(os.path.join(path , "train")):
#            if class_num == '8':
#                continue
            i = i + 1
            # 生成tfrecords
            name_list = os.listdir(os.path.join(path, "train", class_num))
            np.random.shuffle(name_list)
            for img_name in name_list[:100]:
                img_path = path + "/train" + "/" + class_num + "/" + img_name
                image = Image.open(img_path)
                image = image.convert('RGB')
                image = image.resize((200, 200), Image.ANTIALIAS)
                img_np = np.asarray(image)
                # Origin img
                flipped_patch_np = img_np
                flipped_patch = Image.fromarray(np.uint8(flipped_patch_np))
                image = flipped_patch.tobytes()  # 将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }))  # example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  # 序列化为字符串
                # 上下颠倒
                flipped_patch_np = np.flipud(img_np)
                flipped_patch = Image.fromarray(np.uint8(flipped_patch_np))
                image = flipped_patch.tobytes()  # 将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }))  # example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  # 序列化为字符串
                # 左右颠倒
                flipped_patch_np = np.fliplr(img_np)
                flipped_patch = Image.fromarray(np.uint8(flipped_patch_np))
                image = flipped_patch.tobytes()  # 将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }))  # example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  # 序列化为字符串
                # add gaussian noise
                flipped_patch_np = img_np + np.random.normal(0, 10, size=img_np.shape)
                flipped_patch = Image.fromarray(np.uint8(flipped_patch_np))
                image = flipped_patch.tobytes()  # 将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }))  # example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  # 序列化为字符串
                # Rotate patch by a random angle
                no = random.randrange(-180, 180, 30)
                flipped_patch_np = scipy.ndimage.interpolation.rotate(img_np, no, axes=(1, 0),
                                                                      reshape=False, output=None, order=3,
                                                                      mode='constant', cval=0.0, prefilter=False)
                flipped_patch = Image.fromarray(np.uint8(flipped_patch_np))
                image = flipped_patch.tobytes()  # 将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }))  # example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  # 序列化为字符串
    elif "test" in filename:
        name_list = os.listdir(os.path.join(path, "test", class_num))
        for class_num in name_list:
#            if class_num == '8':
#                continue
            i = i + 1
            for img_name in os.listdir(path + "/test" + "/" + class_num):
                img_path = path + "/test" + "/" + class_num + "/" + img_name
                image = Image.open(img_path)
                image = image.resize((200, 200), Image.ANTIALIAS)
                img_np = np.asarray(image)
                # Origin img
                flipped_patch_np = img_np
                flipped_patch = Image.fromarray(np.uint8(flipped_patch_np))
                image = flipped_patch.tobytes()  # 将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }))  # example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()

tfrecords_generator(filename1)
#tfrecords_generator(filename2)