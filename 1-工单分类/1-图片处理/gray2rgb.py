# -- coding: utf-8 --

# 将图片转为RGB格式（防止灰度图）
from wand.image import Image
from wand.display import display
from wand.color import Color
from PIL import Image as PImage
import os
import glob
import numpy as np

SIZE = 500

PATH = os.path.pardir # 父目录
PATH_OUT = os.path.join(PATH, 'convert')
bg = Color('white') # 设置要切换的边缘颜色，此处为白色

for folder  in ['train', 'test', 'val']:
    for img_class in os.listdir(os.path.join(PATH,folder)):
        path_img = os.path.join(PATH, folder,img_class,'*.jpg')
    #    path_out = os.path.join(PATH_OUT, folder)
        names = glob.glob(path_img)
#        np.random.shuffle(names)
        print(folder, img_class)
        for name in names:
#            img = Image(filename=name)
            img = PImage.open(name)
#            img.trim(color=bg, fuzz=20) # 切除白块，20是测试出来的
            img = img.convert('RGB')
#            img.resize(SIZE, SIZE)
#            print(img.info)
#            print(name)
            img.save(name)
