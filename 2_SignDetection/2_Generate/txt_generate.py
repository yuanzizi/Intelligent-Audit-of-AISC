# -*- coding:utf-8 -*-
# 该代码根据已生成的xml，制作VOC数据集中的trainval.txt;train.txt;test.txt和val.txt
# trainval占总数据集的50%，test占总数据集的50%；train占trainval的50%，val占trainval的50%；
# 上面所占百分比可根据自己的数据集修改，如果数据集比较少，test和val可少一些
import os
import random

def txt_generate(xmlfilepath, txtsavepath, trainval_percent, train_percent):
    if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)
    # 数据集大小
    xmlfile = os.listdir(xmlfilepath)
    numofxml = len(xmlfile)

    # trainval及test大小
    random.shuffle(xmlfile)
    trainval = xmlfile[0:round(numofxml * trainval_percent)]
    test = xmlfile[round(numofxml * trainval_percent):numofxml]

    # train及val大小
    numoftrainval = len(trainval)
    random.shuffle(trainval)
    train = trainval[0:round(numoftrainval * train_percent)]
    val = trainval[round(numoftrainval * train_percent):numoftrainval]

    print('writing txt......')

    # 写txt文件
    list_trainval = open('%s/trainval.txt' % txtsavepath,'w')
    list_train = open('%s/train.txt' % txtsavepath,'w')
    list_val = open('%s/val.txt' % txtsavepath,'w')
    list_test = open('%s/test.txt' % txtsavepath,'w')
    '''
    for n in range(numofxml):
        if '%06d.xml' % n in trainval:
            list_trainval.write('%06d\n' % n)
            if '%06d.xml' % n in train:
                list_train.write('%06d\n' % n)
            else:
                list_val.write('%06d\n' % n)
        else:
            list_test.write('%06d\n' % n)
    '''
    for xml in trainval:
        name = os.path.splitext(xml)[0]
        list_trainval.write('%s\n' % name)
        
    for xml in train:
        name = os.path.splitext(xml)[0]
        list_train.write('%s\n' % name)
    
    for xml in val:
        name = os.path.splitext(xml)[0]
        list_val.write('%s\n' % name)
    
    for xml in test:
        name = os.path.splitext(xml)[0]
        list_test.write('%s\n' % name)
        
    list_trainval.close()
    list_train.close()
    list_val.close()
    list_test.close()

    print('Done')
    print('total xml:%d, trainval:%d, test:%d, train:%d, val:%d' % (numofxml, numoftrainval, len(test), len(train), len(val)))

PATH = os.path.join('..','0_Data','sign3')
xml_path = os.path.join(PATH,'Annotations')
txt_path = os.path.join(PATH,'ImageSets','Main')
trainval_percents = 0.8
train_percents = 0.8
txt_generate(xml_path, txt_path, trainval_percents, train_percents)
