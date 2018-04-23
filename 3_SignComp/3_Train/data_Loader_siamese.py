# -- coding:utf-8 --
import tensorflow as tf
import numpy as np
import os
import json
import random
from resize import get_names_img as get_names_img
from resize import get_sign_img as get_sign_img
from PIL import Image

# 客户标准名字图片为由客户姓名生成的名字图片
# 客户签名图片即为客户实际签名图片

# y_out.json 保存客户标准名字图片名字及对应标签
# input:
#   file_path: y_out.json保存路径
# return:
#   temps: 图片名字列表
#   s: 图片标签列表（即客户名字） 
def get_names_list(file_path):
    json_file = os.path.join(file_path, 'y_out.json')
    # 基于ubuntu中文编码库的问题encoding取GB18030
    # window系统下时encoding可选utf-8
    with open(json_file, 'r', encoding="GB18030") as f:
        temp = json.loads(f.read())
        temps = list(temp.keys())
    s = []
    for item in temps:
        s.append(temp[item])
    temps[0] = []
    s[0] = []
    return temps, s

# 由客户签名图片名字获取客户姓名（由实际情况而定）
# input:
#   img_name: 客户签名图片名字
# return:
#   label: 客户名字标签
def get_sign_label(img_name):
    if '_' in img_name[-9:-6]:
        label = img_name[-8:-6]
    else:
        label = img_name[-9:-6]
    return label

# 由客户名字获取客户标准名字图片名字
# input:
#   label: 客户名字标签
#   temps: 图片名字列表
#   s: 图片标签列表（即客户名字） 
# return:
#   返回的为客户标准名字图片的名字
def get_sign_id(label, temps, s):
    words = []
    for word in label:
        words.append(word)
    return temps[s.index(words)][0:-2]

# 读取名字的列表，并分为训练集与测试集，并保存
# input:
#   siamese_path: 签名文件夹sign所在路径
# return:
#   返回训练测试列表
def read_ids(siamese_path):
    sign_list = os.listdir(os.path.join(siamese_path, 'sign'))
    test_list = random.sample(sign_list, 50)
    train_list = list(set(sign_list) - set(test_list))

    file1 = open(siamese_path + '/train_index.txt', 'w')
    total_train_list_line = [train_line + '\n' for train_line in train_list]
    file1.writelines(total_train_list_line)
    file1.close()
    file2 = open(siamese_path + '/test_index.txt', 'w')
    total_test_list_line = [test_line + '\n' for test_line in test_list]
    file2.writelines(total_test_list_line)
    file2.close()

    return train_list, test_list

# 获取batchsize大小的训练图片
def get_train_img(siamese_path, batch_size, false_ratio=0):
    # 训练列表读取
    if os.path.exists(os.path.join(siamese_path, 'train_index.txt')):
        file = open(os.path.join(siamese_path, 'train_index.txt'), 'r')
        train_list = file.read().encode('utf-8').decode('utf-8-sig').split()
        file.close()
    else:
        train_list, _ = read_ids(siamese_path)
    ids, names = get_names_list(os.path.join(siamese_path, 'names'))
    train_batch = random.sample(train_list, batch_size)
    
    sign_imgs = []
    name_imgs = []
    labels = []
    for batch in range(batch_size):
        sign_name = train_batch[batch]
        sign_label = get_sign_label(sign_name)
        names_id = get_sign_id(sign_label, ids, names)
        # 客户标准名字图片中每个客户有六种字体签名，随机选取
        names_name = names_id + '_' + str(random.randint(0, 5)) + '.tif'
        # 训练中选取一对正样本一对负样本
        # 正样本即客户签名图片与客户标准名字图片选取为同一人的，标签定为1
        # 正样本即客户签名图片与客户标准名字图片选取为不同人的，标签定为0
        sign_img = get_sign_img(os.path.join(siamese_path, 'sign', sign_name))
        name_img = get_names_img(os.path.join(siamese_path, 'names', names_name))
        sign_imgs.append(sign_img)
        name_imgs.append(name_img)
        labels.append('1')
        if false_ratio >= 1:
            for false_i in range(false_ratio): 
                false_name_id = random.randint(1, 1024)
                while false_name_id == names_id:
                    false_name_id = random.randint(1, 1024)
                false_name = str(false_name_id) + '_' + str(random.randint(0, 5)) + '.tif'
                false_img = get_names_img(os.path.join(siamese_path, 'names', false_name))
                sign_imgs.append(sign_img)
                name_imgs.append(false_img)
                labels.append('0')
        else:
            if random.random() > false_ratio:
                sign_imgs.append(sign_img)
                name_imgs.append(false_img)
                labels.append('0')
    sign_imgs = np.array(sign_imgs)
    name_imgs = np.array(name_imgs)
    labels = np.array(labels)
    sign_imgs = sign_imgs[:, :, :, np.newaxis]
    name_imgs = name_imgs[:, :, :, np.newaxis]
    return sign_imgs, name_imgs, labels

# 获取训练图片
# num为测试列表中测试图片序号
def get_test_img(siamese_path, num):
    if os.path.exists(os.path.join(siamese_path, 'test_index.txt')):
        file = open(os.path.join(siamese_path, 'test_index.txt'), 'r')
        # .encode('utf-8').decode('utf-8-sig')为ubuntu中文编码库问题需要，window可删去
        test_list = file.read().encode('utf-8').decode('utf-8-sig').split()
        file.close()
    else:
        _, test_list = read_ids(siamese_path)
    ids, names = get_names_list(os.path.join(siamese_path, 'names'))
    # train_batch = random.sample(train_list, batch_size)
    
    sign_imgs = []
    name_imgs = []
    labels = []
    
    sign_name = test_list[num]
    sign_label = get_sign_label(sign_name)
    names_id = get_sign_id(sign_label, ids, names)
    names_name = names_id + '_' + str(random.randint(0, 5)) + '.tif'
    false_name_id = random.randint(1, 1024)
    while false_name_id == names_id:
        false_name_id = random.randint(1, 1024)
    false_name = str(false_name_id) + '_' + str(random.randint(0, 5)) + '.tif'
    # print(os.path.join(siamese_path, 'sign', sign_name))
    # print(sign_name)
    sign_img = get_sign_img(siamese_path + '/sign/' + sign_name)
    name_img = get_names_img(os.path.join(siamese_path, 'names', names_name))
    false_img = get_names_img(os.path.join(siamese_path, 'names', false_name))
    # Image._show(Image.fromarray(sign_img*255))
    # Image._show(Image.fromarray(name_img*255))
    # Image._show(Image.fromarray(false_img*255))
    sign_imgs.append(sign_img)
    name_imgs.append(name_img)
    labels.append('1')
    sign_imgs.append(sign_img)
    name_imgs.append(false_img)
    labels.append('0')
    
    sign_imgs = np.array(sign_imgs)
    name_imgs = np.array(name_imgs)
    labels = np.array(labels)
    sign_imgs = sign_imgs[:, :, :, np.newaxis]
    name_imgs = name_imgs[:, :, :, np.newaxis]
    return sign_imgs, name_imgs, labels


if __name__ == '__main__':
    s, n, l = get_train_img('/data1/data_greebear/data/siamese', 32)
    print(s.shape)
    print(n.shape)
    print(l.shape)