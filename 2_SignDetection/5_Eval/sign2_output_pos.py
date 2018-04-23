
# coding: utf-8


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import os
import cv2
import time
# import argparse
# import multiprocessing
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import glob
import pandas as pd
import PIL.Image as Image
import json
import sys

tf.reset_default_graph() 

PATH_MODEL = os.path.join(os.path.pardir,'6_Model','sign2')
PATH_SET = os.path.join('..','0_Data','sign2','ImageSets','main')

PATH_TO_CKPT = os.path.join( os.path.join(PATH_MODEL,'frozen_inference_graph.pb'))
#PATH_TO_INFO = os.path.join( os.path.join(PATH_MODEL,'job_info.csv'))
PATH_TO_LABELS = os.path.join(os.path.join(PATH_MODEL, '5_label_map.pbtxt'))
PATH_TO_IMG = os.path.join('..','0_Data','sign2','JPEGImages')
PATH_PRED = os.path.join('..','0_Data','sign2','mAP','predicted')
NUM_CLASSES = 2
IMAGE_SIZE = (18, 18)

# 检查路径是否都存在
for path in [PATH_MODEL, PATH_TO_CKPT, PATH_TO_LABELS, PATH_SET,PATH_PRED]:
    if not os.path.exists(path):
        print(path,'do not exist!')
        sys.exit()
        
# 输出图片的数量       
img_list = glob.glob(os.path.join(PATH_TO_IMG, '*.jpg'))
print('num of imgs=',len(img_list))
img_names = [os.path.split(i)[1] for i in img_list]

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def read_text(name = 'test.txt'):
    
    names = []
    import codecs
    path = os.path.join(PATH_SET, name)
    if not os.path.exists(path):
        print('path does not exist!', path)
        return
    with codecs.open(path, encoding='utf-8', mode='rU') as f:
        for idx, line in enumerate(f):
            names.append(line.strip())
            if idx > 99999:
                break
    return names
    
test_names = read_text()
print('num fo test', len(test_names))



#Load a frozen TF model 
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')




data = []
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for idx,name in enumerate(test_names[:]):
            if idx % 10 == 0:
                print('processing', idx)
            if name + '.jpg' not in img_names:
                print(name,'not in xml_names')
                continue
            image_path = os.path.join(PATH_TO_IMG,name+'.jpg')
#        for image_path in PATH_TO_IMG[:5]:
            image = Image.open(image_path) # .resize((300,300))
            image_np = np.array(image).astype(np.uint8)
            image_np = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np})

            # 将数据存储成json格式
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)
            # 只要类别1，且阈值大于0.5
            cond = np.column_stack([classes == 1, scores > 0.5]).all(axis=1)
            with open(os.path.join(PATH_PRED, name+'.txt'), 'w') as new_f:
                if cond.sum() > 0: # 识别出来的目标大于0
                    for box, pred_class, score in zip(boxes[cond], classes[cond], scores[cond]):
                        label = category_index[pred_class]['name']
                        points = (box* ([500]*4)).astype(np.int)
                        points = points[[1,0,3,2]]
#                        print(name, points, label, score)
                        new_f.write(label + " " + str(score) + " " + str(points[0]) + " " + str(points[1]) + " " + str(points[2]) + " " + str(points[3]) + '\n')
                else: # 识别出来的目标等于0，则输出一个默认值
                    print(name,'have no object')
                    new_f.write(label + " " + str(1.0) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + '\n')
            '''
            serial = img_name.split('_')[0]
            try:
                name = info[info.serial_num == serial]['name'].iloc[0]
            except:
                print(serial)
                continue
            if boxes[cond].shape[0] == 0:
                points = []*4
            else:
                points = (boxes[cond][0]* ([500]*4)).astype(np.int)
                points = points[[1,0,3,2]]
            to_json = {}
            for idx,p in enumerate(['xmin','ymin','xmax','ymax']):
                if len(points) == 4:
                    to_json[p] = int(points[idx])
                else:
                    to_json[p] = ''
            to_json['name'] = name
            to_json = json.dumps(to_json,ensure_ascii=False)
            data.append((img_name, to_json))
#             print(img_name, points,name)


out = pd.DataFrame(data,columns=['img_name','data'])
out.head()




out.to_csv('sign_detection_json.csv',index=False)
'''