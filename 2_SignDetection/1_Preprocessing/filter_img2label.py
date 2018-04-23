# -- coding: utf-8 --

# 针对两种保证书，将概率低于阈值的图片删除掉

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
from sklearn.metrics import confusion_matrix, accuracy_score

SIZE = 200
PATH_DATA = os.path.join('..','..','img4label') # 存放图片的地址
PATH_OUT = os.path.join('..','..','img4label')
PATH_MODEL = os.path.join('..','..','model1')
RECORD_NAME = os.path.join(PATH_DATA,'img4label.tfrecord') # 储存转换后的图片，预测后可以删除


def getFolders(is_shuffle = True):
    
    folders = os.listdir(PATH_DATA) # 获取所有文件
    
    def filterFolder(f): # 判断文件是否文件夹
        if os.path.splitext(f)[1] == '' and f != 'model' and f != 'TEMP' \
        and f != 'BACKUP' and f != 'OUTPUT' and f != '__pycache__' \
        and f != 'code':
            return True # 如果是文件夹
        return False
    
    folders = list(filter(filterFolder, folders))
    if is_shuffle:
        np.random.seed(2017)
        np.random.shuffle(folders)
    return folders

def write_tfrecord(names, size=200): # 将文件夹下的图片写入tfrecord
    
    bg = Color('white') # 设置要切换的边缘颜色，此处为白色

    with tf.python_io.TFRecordWriter(RECORD_NAME) as writer:
    
        for name in names:
            img = WImage(filename=name)
            img.trim(color=bg, fuzz=20) # 切除白块，20是测试出来的
            img.resize(SIZE, SIZE)
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

def read_copy():
    
    dataset = tf.data.TFRecordDataset(RECORD_NAME)
    dataset = dataset.map(read_decode)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(5)
    iterator =  dataset.make_one_shot_iterator()
    
    batch_image, batch_name = iterator.get_next()
    score_label = model(batch_image, False)
    pred_label = tf.argmax(score_label, 1)
    softmax = tf.nn.softmax(score_label)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    saver = tf.train.Saver()
    y_true, y_pred, names, label_prob, label_score = [], [], [], [], []
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, os.path.join(PATH_MODEL,'model1.ckpt')) 

        while True:
            try:
                label_pred,name,label_prob = sess.run([pred_label, batch_name,softmax])
                name = list(map(decode_name, name))
#                print(label_pred, name)
                label_prob = np.array(label_prob)
                print(label_prob.shape)
                for p,n,s in zip(label_pred, name,label_prob.max(axis=1)):
                    if s < 0.95: # 如果准确率低于阈值，则移除掉该图片
                        print(p,n,s)
                        out_n = os.path.join(PATH_OUT,'other',os.path.split(n)[1])
                        shutil.copyfile(n,out_n) # 复制到其他文件夹
                        os.remove(n) # 删除图片
            except Exception as e:
                print('Finish')
#                print(e)
                break
#    return y_true, y_pred, names, label_prob,label_score

def filterImg(name):
   
    bg = Color('white') # 设置要切换的边缘颜色，此处为白色
    img = WImage(filename=name)  
    h1 = img.height
    w1 = img.width
    if h1* w1 < 10000: # 过滤掉移动的Logo
        return False    
    img.trim(color=bg, fuzz=20) # 切除白块，20是测试出来的
    h2 = img.height
    w2 = img.width
    if (w2*h2 / (w1*h1) < 0.6) and h1 * w1 > 3000000: # 剔除掉有水印的图片
        return False
    return True


folders = getFolders(is_shuffle=True)
folders = folders[:1000]
print(folders)

names = []
for idx,folder in enumerate(folders[:5]):
#    print(folder)
    if folder not in ['5','7']:
        continue
    img_list = glob.glob(os.path.join(PATH_DATA,folder, '*.jpg'))
    names.extend(img_list)
print(len(names))
#names = list(filter(filterImg, names))
#print(len(names))

def generateImgList(names, num=10): # 每次生成包含num个文件名的列表
    idx = 0
    while idx < len(names):
        yield names[idx:idx+num]
        idx += num

for idx, name_list in enumerate(generateImgList(names[:],50)):
#    print(name_list)
    print('idx',idx)
    write_tfrecord(name_list)
    tf.reset_default_graph() 
    read_copy()
