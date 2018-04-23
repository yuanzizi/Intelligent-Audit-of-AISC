import os
from PIL import Image
import glob
import numpy as np

'''
for idx in os.listdir("./5"):
    img = Image.open("./5/"+idx)
    h = img.size[0]
    w = img.size[1]
    img = img.crop((int(w/7), int(h*2/5), int(w*3/5), int(h*7/10))).resize((500, 500))
    img.save("./out_5/5_%s.jpg" % idx)

for idx in os.listdir("./7"):
    img = Image.open("./7/"+idx)
    w = img.size[0]
    h = img.size[1]
    img = img.crop((int(w/2), int(h*5/16), int(w*9/10), int(h*11/16))).resize((500, 500))
    img.save("./out_7/7_%s.jpg"%idx)
    
'''

PATH = os.path.pardir



#imgs = glob.glob(os.path.join(PATH,'5', '*.jpg'))
imgs = glob.glob(os.path.join(PATH,'7', '*.jpg'))

np.random.shuffle(imgs)


# 针对类型5
'''
for idx, name in enumerate(imgs[:]):
    img = Image.open(name)
    h = img.size[0]
    w = img.size[1]
    img = img.crop((int(w/6), int(h/3), int(w*0.5), int(h*0.78))).resize((500, 500))
    img.save(os.path.join(PATH,'5_crop',"%06d.jpg" % idx))
'''

# 针对类型7
for idx, name in enumerate(imgs[:]):
    img = Image.open(name)
#    print(img.size)
    w = img.size[0]
    h = img.size[1]
    img = img.crop((int(w*0.5), int(h*0.3), int(w*0.95), int(h*0.8)))
    img = img.resize((500, 500))
#    img = img.crop((int(0), int(0), int(w), int(h))).resize((500, 500))
#    img = img.crop((int(w*0.5), int(h/3), int(w*0.9), int(h*0.78))).resize((500, 500))
    img.save(os.path.join(PATH,'7_crop',"%06d.jpg" % (idx+300)))
	
	
# 针对打印版
'''
for img_path in img_paths[:]:
    
    
    img = PImage.open(img_path)
#    print(img.size)
    w = img.size[0]
    h = img.size[1]
    img = img.crop((int(w*0.22), int(h*0.35), int(w*0.47), int(h*0.6)))
    img = img.resize((500, 500))
#    img.show()
#    img = img.crop((int(0), int(0), int(w), int(h))).resize((500, 500))
#    img = img.crop((int(w*0.5), int(h/3), int(w*0.9), int(h*0.78))).resize((500, 500))
    img.save(os.path.join(PATH_OUT,os.path.split(img_path)[1]))
'''