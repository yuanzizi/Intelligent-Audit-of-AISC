# -*-coding:utf-8-*-

# 将有水印的图片过滤掉

from wand.image import Image as WImage
from wand.color import Color
import os
import glob
import numpy as np

PATH = os.path.join(os.path.pardir, '201801_JPG')
PATH_OUT = os.path.join(os.path.pardir, '201801_JPG_OUT')
SIZE = 300

if not os.path.exists(PATH_OUT):
    os.mkdir(PATH_OUT)

folders = os.listdir(PATH)
np.random.shuffle(folders)
bg = Color('white') # 设置要切换的边缘颜色，此处为白色

print('num of folders', len(folders))

for idx, folder in enumerate(folders[:5]):
    
    if idx % 100 == 0 :
        print('process', idx)
    
    PATH_FROM = os.path.join(PATH, folder)
    PATH_TO = os.path.join(PATH_OUT, folder)
    if not os.path.exists(PATH_TO):
        os.mkdir(PATH_TO)
    imgs = glob.glob(os.path.join(PATH_FROM, '*.jpg'))
    
    for name in imgs:
        
        img = WImage(filename=name)  
        h1 = img.height
        w1 = img.width
        if h1* w1 < 10000:
            continue
        
        img.trim(color=bg, fuzz=20) # 切除白块，20是测试出来的
        h2 = img.height
        w2 = img.width
        
        if (w2*h2 / (w1*h1) < 0.6) and h1 * w1 > 3000000:
            continue
        img.resize(SIZE, SIZE)

        img.save(filename=os.path.join(PATH_TO,folder + '_' + os.path.split(name)[1]))
        
    