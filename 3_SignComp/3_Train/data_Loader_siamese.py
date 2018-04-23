# -- coding:utf-8 --
import tensorflow as tf
import numpy as np
import os
import json
import random
from resize import get_names_img as get_names_img
from resize import get_sign_img as get_sign_img
from PIL import Image

# �ͻ���׼����ͼƬΪ�ɿͻ��������ɵ�����ͼƬ
# �ͻ�ǩ��ͼƬ��Ϊ�ͻ�ʵ��ǩ��ͼƬ

# y_out.json ����ͻ���׼����ͼƬ���ּ���Ӧ��ǩ
# input:
#   file_path: y_out.json����·��
# return:
#   temps: ͼƬ�����б�
#   s: ͼƬ��ǩ�б����ͻ����֣� 
def get_names_list(file_path):
    json_file = os.path.join(file_path, 'y_out.json')
    # ����ubuntu���ı���������encodingȡGB18030
    # windowϵͳ��ʱencoding��ѡutf-8
    with open(json_file, 'r', encoding="GB18030") as f:
        temp = json.loads(f.read())
        temps = list(temp.keys())
    s = []
    for item in temps:
        s.append(temp[item])
    temps[0] = []
    s[0] = []
    return temps, s

# �ɿͻ�ǩ��ͼƬ���ֻ�ȡ�ͻ���������ʵ�����������
# input:
#   img_name: �ͻ�ǩ��ͼƬ����
# return:
#   label: �ͻ����ֱ�ǩ
def get_sign_label(img_name):
    if '_' in img_name[-9:-6]:
        label = img_name[-8:-6]
    else:
        label = img_name[-9:-6]
    return label

# �ɿͻ����ֻ�ȡ�ͻ���׼����ͼƬ����
# input:
#   label: �ͻ����ֱ�ǩ
#   temps: ͼƬ�����б�
#   s: ͼƬ��ǩ�б����ͻ����֣� 
# return:
#   ���ص�Ϊ�ͻ���׼����ͼƬ������
def get_sign_id(label, temps, s):
    words = []
    for word in label:
        words.append(word)
    return temps[s.index(words)][0:-2]

# ��ȡ���ֵ��б�����Ϊѵ��������Լ���������
# input:
#   siamese_path: ǩ���ļ���sign����·��
# return:
#   ����ѵ�������б�
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

# ��ȡbatchsize��С��ѵ��ͼƬ
def get_train_img(siamese_path, batch_size, false_ratio=0):
    # ѵ���б��ȡ
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
        # �ͻ���׼����ͼƬ��ÿ���ͻ�����������ǩ�������ѡȡ
        names_name = names_id + '_' + str(random.randint(0, 5)) + '.tif'
        # ѵ����ѡȡһ��������һ�Ը�����
        # ���������ͻ�ǩ��ͼƬ��ͻ���׼����ͼƬѡȡΪͬһ�˵ģ���ǩ��Ϊ1
        # ���������ͻ�ǩ��ͼƬ��ͻ���׼����ͼƬѡȡΪ��ͬ�˵ģ���ǩ��Ϊ0
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

# ��ȡѵ��ͼƬ
# numΪ�����б��в���ͼƬ���
def get_test_img(siamese_path, num):
    if os.path.exists(os.path.join(siamese_path, 'test_index.txt')):
        file = open(os.path.join(siamese_path, 'test_index.txt'), 'r')
        # .encode('utf-8').decode('utf-8-sig')Ϊubuntu���ı����������Ҫ��window��ɾȥ
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