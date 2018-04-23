# -- coding: utf-8 --
'''
文件夹结构
| ---- 13411978121020180131174754151
| ---- 13411964458020180131180551058
| ---- ...其他包含切片的工单文件夹
| ---- TEMP（自动生成，临时储存转换后的图片，可以删除）
| ---- OUTPUT（自动生成，储存分类后的原始图片）
|    | ---- 0
|    | ---- 1
|    | ---- ...（共有十个文件夹，第九和十个文件夹暂无用处）
| ---- model（储存模型权重文件）
|    | ---- model1.ckpt.data-00000-of-00001
|    | ---- model1.ckpt.index
|    | ---- model1.ckpt.meta
|    | ---- checkpoint
| ---- code（储存代码）
|    | ---- main_class.py（主文件）
|    | ---- model1.py（模型的结构文件）
|    | ---- ...
'''

import tensorflow as tf
import glob
import os
import io
import numpy as np
from model1_0320 import model
#from  PIL import Image
import shutil
from wand.image import Image
from wand.display import display
from wand.color import Color
from PIL import Image as PImage
from sklearn.metrics import confusion_matrix, accuracy_score

SIZE = 200
PATH_DATA = os.path.join(os.path.pardir,'data')
FOLDER_TYPE = 'val'
RECORD_NAME = os.path.join(PATH_DATA,'%s.tfrecord'%FOLDER_TYPE) # 储存转换后的图片，预测后可以删除
#path = 
#RECORD_NAME = os.paht.join(path,'model1_test.tfrecords')

def getFolders():
    
    folders = os.listdir(os.path.join(PATH_DATA,FOLDER_TYPE)) # 获取所有文件
    
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


def write_tfrecord(all_names): # 将文件夹下的图片写入tfrecord
    
    with tf.python_io.TFRecordWriter(RECORD_NAME) as writer:
        for label_true, names in all_names.items():
            for name in names:
                byte_img = PImage.open(name)
                byte_img = byte_img.convert('RGB') # 有些图片是黑白灰度，需要转换成RGB
                byte_img = byte_img.resize((200,200))
                byte_img = byte_img.tobytes()
                byte_name = bytes(name, encoding='utf-8')
#                print('write:',label_true, byte_name, len(byte_img))
                tf_feature = {'byte_img':tf.train.Feature(bytes_list=tf.train.BytesList(value=[byte_img])),
                              'byte_name':tf.train.Feature(bytes_list=tf.train.BytesList(value=[byte_name])),
                              'label_true':tf.train.Feature(int64_list=tf.train.Int64List(value=[label_true]))}   
                tf_features = tf.train.Features(feature=tf_feature)
                example = tf.train.Example(features=tf_features)
                writer.write(example.SerializeToString())    

def read_decode(serialized_example):

    tf_features = {'byte_img':tf.FixedLenFeature([], tf.string),
                   'byte_name':tf.FixedLenFeature([], tf.string),
                   'label_true':tf.FixedLenFeature([], tf.int64)}
    
    features = tf.parse_single_example(
            serialized_example,
            features=tf_features)
    
    img = tf.decode_raw(features['byte_img'], tf.uint8)
    img = 2*tf.cast(img, tf.float32) * (1./255) -1
    img = tf.reshape(img, [200,200,3])
    name = tf.cast(features['byte_name'],tf.string)
    label_true = tf.cast(features['label_true'], tf.int64)
    return img, name, label_true
       
def read_copy():
    
    if not os.path.exists(os.path.join(os.path.pardir,'OUTPUT')):
        os.mkdir(os.path.join(os.path.pardir,'OUTPUT'))
    
    for i in range(10): # 检查输出文件夹是否存在
        if not os.path.exists(os.path.join(os.path.pardir,'OUTPUT',str(i))):
            os.mkdir(os.path.join(os.path.pardir,'OUTPUT',str(i)))
    dataset = tf.data.TFRecordDataset(RECORD_NAME)
    dataset = dataset.map(read_decode)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(5)
    iterator =  dataset.make_one_shot_iterator()
    
    batch_image, batch_name, batch_label = iterator.get_next()
    score_label = model(batch_image, False)
    pred_label = tf.argmax(score_label, 1)
    softmax = tf.nn.softmax(score_label)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    saver = tf.train.Saver()
    y_true, y_pred, names, label_prob, label_score = [], [], [], [], []
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, os.path.join(os.path.pardir,'model','model1.ckpt')) 

        while True:
            try:
                label_pred,name,label_true, pred_prob, pred_score = sess.run([pred_label, batch_name, batch_label, softmax, score_label])

#                name = str(name[0],encoding='utf8')
                print(label_pred, label_true)
                y_true.extend(label_true)
                y_pred.extend(label_pred)
                names.extend(name)
                label_prob.extend(pred_prob)
                label_score.extend(pred_score)

#                n = os.path.split(name)
#                shutil.copyfile(name,os.path.join(os.path.pardir,'OUTPUT',str(label), '%s_%s'%(n[0][3:], n[1])))
            except:
                print('Finish')
                break
    return y_true, y_pred, names, label_prob,label_score

folders = getFolders()
img_names = {}
for idx,folder in enumerate(folders[:]):
#    print(folder)
    img_list = glob.glob(os.path.join(PATH_DATA,FOLDER_TYPE,folder, '*.jpg'))
    np.random.shuffle(img_list)
    if folder in ['6','7','8']:
#        continue
        img_names[int(folder)] = img_list[:]
    else:
        img_names[int(folder)] = img_list[:]
#        pass

tf.reset_default_graph() 
    

write_tfrecord(img_names)
y_true, y_pred, names, label_prob, label_score = read_copy()

def decode_name(name):
    return str(name,encoding='utf8')


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

names = list(map(decode_name, names))


y_true = np.array(y_true)
y_pred = np.array(y_pred)
names = np.array(names)
label_prob = np.array(label_prob)
label_score = np.array(label_score)

#label_prob = softmax(label_prob)
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
y_pred2 = y_pred
y_pred[label_prob.max(axis=1) < 0.90] = 8

print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
for t,p,n in zip(y_true[y_true != y_pred], y_pred[y_true != y_pred], names[y_true != y_pred]):
    print(t, p, n)