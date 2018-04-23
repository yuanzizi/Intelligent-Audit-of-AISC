# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import random
import matplotlib.image as mpimg
import numpy as np
import json
import csv
# input_images_path客户签名对应图片路径
input_images_path = 'test_4.12'
# output_images_path对应生成系统名字图片路径
output_images_path = 'test_names'
# words_path单个汉字图片路径
words_path = 'single_word_generate - 副本/images'
# ids选择生成字体序号（与单个汉字图片中字体相对应）
ids = [0, 1, 2, 3, 4, 5]


def convert_dict_to_list(dict_file):
    with open(dict_file, 'r') as f:
        temp = json.loads(f.read())
        tems = list(temp.keys())
        s = []
        for i in range(len(tems)):
            s.append(temp[tems[i]])
    return s


def read_csv(csv_file):
    csvFile = open(csv_file, "r", encoding='UTF-8')
    reader = csv.reader(csvFile)
    data = []
    for item in reader:
        data.append(item)
    csvFile.close()
    return data


def generate(input_path, output_path):
    # json 汉字列表
    json_path = os.path.join(input_path, 'y_tag.json')
    word_lists = convert_dict_to_list(json_path)
    labels = {}

    # csv 姓名
    csv_File = os.path.join(input_path, 'conver_list.csv')
    name_lists = read_csv(csv_File)
    print(name_lists)

    all_list = os.listdir(os.path.join(input_path, '0'))
    for i in range(1, len(name_lists)-1):
        for id in ids:
            name_all = mpimg.imread(os.path.join(input_path, '0', '0_NotoSansHans-DemiLight.otf.jpg'))
            label = []
            for name_list in name_lists[i][1]:
                if name_list == ' ':
                    continue
                try:
                    word = word_lists.index(name_list)
                except:
                    continue
                name_id = os.path.join(input_path, str(word), all_list[id])
                name_img = mpimg.imread(name_id)
                name_all = np.concatenate([name_all, name_img], axis=-1)
                label.append(name_list)
            name_all = np.array(name_all)
            labels[str(i)+'_'+str(id)] = label

            output_image = os.path.join(output_path, str(i)+'_'+str(id)+'.tif')
            mpimg.imsave(output_image, name_all[:, 64:], cmap='Greys')
    output_json = os.path.join(output_path, 'y_out.json')
    with open(output_json, 'w') as o:
        json.dump(labels, o, ensure_ascii=False)

######################################################################################
######################################################################################
####################################version 2#########################################
# 读取文件中图片名字并返回图片sign标签
def read_filesign(path):
    jpglists = os.listdir(path)
    jpgsigns = [jpglist.split('_')[1] for jpglist in jpglists]

    return jpgsigns


def generate_v2(input_path, word_path, output_path):
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    json_path = os.path.join(word_path, 'y_tag.json')
    word_lists = convert_dict_to_list(json_path)
    labels = {}

    name_lists = read_filesign(input_path)

    # 每个汉字下字体名字列表
    all_list = os.listdir(os.path.join(word_path, '0'))
    for i in range(len(name_lists)):
        for id in ids:
            name_all = []
            label = []
            for name_list in name_lists[i]:
                if ' ' in name_list:
                    print(name_list)
                    continue
                try:
                    word = word_lists.index(name_list)
                except:
                    continue
                name_id = os.path.join(word_path, str(word), all_list[id])
                name_img = mpimg.imread(name_id)
                if len(name_all) == 0:
                    name_all = name_img
                else:
                    name_all = np.concatenate([name_all, name_img], axis=-1)
                label.append(name_list)
            name_all = np.array(name_all)
            labels[str(i) + '_' + str(id)] = label

            output_image = os.path.join(output_path, str(i) + '_' + str(id) + '.tif')
            mpimg.imsave(output_image, name_all, cmap='Greys')
    output_json = os.path.join(output_path, 'y_out.json')
    with open(output_json, 'w') as o:
        json.dump(labels, o, ensure_ascii=False)


if __name__ == "__main__":
    # generate(input_images_path, output_images_path)
    """
    tem = convert_dict_to_list()
    tems = list(tem.keys())
    s = []
    for i in range(len(tems)):
        s.append(tem[tems[i]])
    print(s)
    print(s.index('全'))
    """
    generate_v2(input_images_path, words_path, output_images_path)
