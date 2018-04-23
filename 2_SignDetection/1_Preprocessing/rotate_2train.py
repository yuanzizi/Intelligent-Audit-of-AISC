# -- coding: utf-8 --

import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import os
PATH = os.path.join(os.getcwd())

img_paths = glob.glob(os.path.join(PATH,'1','*.jpg'))

for path in img_paths:
    
    name = os.path.split(path)[1]
    img = Image.open(path)
    
#    img_0 = img.transpose(Image.ROTATE_90)
#    img_0.save(os.path.join(PATH,'0',name))
    
    img_2 = img.transpose(Image.ROTATE_270)
    img_2.save(os.path.join(PATH,'2',name))
    
    img_3 = img.transpose(Image.ROTATE_180)
    img_3.save(os.path.join(PATH,'3',name))