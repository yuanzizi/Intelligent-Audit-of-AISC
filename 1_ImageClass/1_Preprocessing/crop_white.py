# -- coding: utf-8 --

# 切除白块

from wand.image import Image
from wand.display import display
from wand.color import Color
import os
import glob
import numpy as np



SIZE = 500

PATH = os.path.pardir # 父目录
PATH_OUT = os.path.join(PATH, 'convert')
bg = Color('white') # 设置要切换的边缘颜色，此处为白色

for folder  in ['0', '1', '2']:
    path_img = os.path.join(PATH, folder, '*.jpg')
    path_out = os.path.join(PATH_OUT, folder)
    names = glob.glob(path_img)
    np.random.shuffle(names)
    for name in names:
        img = Image(filename=name)
        img.trim(color=bg, fuzz=20) # 切除白块，20是测试出来的
        img.resize(SIZE, SIZE)
        img.save(filename=os.path.join(path_out, os.path.split(name)[1]))
