# -- coding: utf-8 --

# 利用模型对工单图片进行分类，并且保存

import tensorflow as tf
import glob
import os
import io
import numpy as np
from model1_0315 import model
#from  PIL import Image
import shutil
from wand.image import Image
from wand.display import display
from wand.color import Color
from PIL import Image as PImage
from sklearn.metrics import confusion_matrix, accuracy_score

SIZE = 200
PATH_DATA = os.path.join(os.path.pardir,'data')
PATH_OUT = os.path.join(os.path.pardir,'split')
#FOLDER_TYPE = ''
RECORD_NAME = os.path.join(os.path.pardir,'split.tfrecord') # 储存转换后的图片，预测后可以删除
#path = 
#RECORD_NAME = os.paht.join(path,'model1_test.tfrecords')

def getFolders():
    
    folders = os.listdir(PATH_DATA) # 获取所有文件
    
    def filterFolder(f): # 判断文件是否文件夹
        if os.path.splitext(f)[1] == '' and f != 'model' and f != 'TEMP' \
        and f != 'BACKUP' and f != 'OUTPUT' and f != '__pycache__' \
        and f != 'code':
            return True # 如果是文件夹
        return False
    
    folders = list(filter(filterFolder, folders))
    return folders



def convertImg(folder): # 切除白块和重置尺寸
    
    if not os.path.exists(os.path.join(PATH_DATA,'TEMP')):
        os.mkdir(os.path.join(PATH_DATA,'TEMP'))
        
    temp_names = glob.glob(os.path.join(PATH_DATA,'TEMP','*.*'))
    list(map(lambda x: os.remove(x), temp_names)) # 把temp文件夹里面的文件清空
    
    names = glob.glob(os.path.join(PATH_DATA,FOLDER_TYPE,folder, '*.jpg'))
    for idx,name in enumerate(names):
        img = Image(filename=name)  
        bg = Color('white') # 设置要切换的边缘颜色，此处为白色
        img.trim(color=bg, fuzz=20) # 切除白块，20是测试出来的
        img.resize(SIZE, SIZE)
#        img.background_color = bg
#        img.format        = 'jpg'
#        img.alpha_channel = False
        img.save(filename=os.path.join(os.path.pardir,'data','TEMP','%s.jpg' % idx))
    return names


def write_tfrecord(names): # 将文件夹下的图片写入tfrecord
    
    with tf.python_io.TFRecordWriter(RECORD_NAME) as writer:
        for name in names:
            byte_img = PImage.open(name)
            byte_img = byte_img.convert('RGB') # 有些图片是黑白灰度，需要转换成RGB
            byte_img = byte_img.resize((200,200))
            byte_img = byte_img.tobytes()
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
       
def read_copy(folder_type):
    
    if not os.path.exists(PATH_OUT):
        os.mkdir(PATH_OUT)
    if not os.path.exists(os.path.join(PATH_OUT, folder_type)):
        os.mkdir(os.path.join(PATH_OUT, folder_type))
    for i in range(9): # 检查输出文件夹是否存在
        if not os.path.exists(os.path.join(PATH_OUT, folder_type,str(i))):
            os.mkdir(os.path.join(PATH_OUT, folder_type,str(i)))
    dataset = tf.data.TFRecordDataset(RECORD_NAME)
    dataset = dataset.map(read_decode)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(5)
    iterator =  dataset.make_one_shot_iterator()
    
    batch_image, batch_name = iterator.get_next()
    batch_score = model(batch_image, False)
    batch_pred = tf.argmax(batch_score, 1)
    batch_softmax = tf.nn.softmax(batch_score)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    saver = tf.train.Saver()
    names, scores, preds, softmaxs = [], [], [], []
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, os.path.join(os.path.pardir,'model','model1.ckpt')) 

        while True:
            try:
                name, pred, score, softmax = sess.run([batch_name,
                                                       batch_pred,
                                                       batch_score,
                                                       batch_softmax])

                name = list(map(lambda x: str(x, encoding='utf8'),name))
                print(pred)
                for p,n in zip(pred,name):
#                    n = os.path.split(name)
                    shutil.copyfile(n, 
                                    os.path.join(PATH_OUT,
                                                 folder_type,
                                                 str(p),os.path.split(n)[1])
                                    )
            except Exception as e:
                print('Finish')
#                print(e)
                break
#    return y_true, y_pred, names, label_prob,label_score

folders = getFolders()
np.random.seed(2018)
np.random.shuffle(folders)
folders_len = len(folders)
folders_type = {}
#folders_type['train'] = folders[:int(folders_len * 0.5)]
#folders_type['test'] = folders[int(folders_len * 0.5):int(folders_len * 0.7)]
folders_type['val'] = folders[int(folders_len * 0.7):]

for folder_type, folder_list in folders_type.items():
    names = []
    print(folder_type, len(folder_list))
    if folder_type != 'train':
        pass
    for folder in folder_list[:]:
#        print(folder)
        names.extend( glob.glob(os.path.join(PATH_DATA,folder,'*.jpg')) )
    tf.reset_default_graph() 
    write_tfrecord(names)
    read_copy(folder_type)
    os.remove(RECORD_NAME)