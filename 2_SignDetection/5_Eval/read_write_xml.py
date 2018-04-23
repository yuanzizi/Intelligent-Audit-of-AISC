# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import glob
import os
import PIL.Image as Image
import PIL.ImageDraw as Draw

for xml_file in glob.glob('0*.xml'):
    print(os.path.split(xml_file)[1])
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
#        print(member[0].text)
        img_name = root.find('filename').text
        points = tuple([int(member[4][i].text) for i in range(4)])
#        print(points)
#        print(img_name)
        img = Image.open(img_name)
        draw = Draw.Draw(img)
#        draw.rectangle(points,outline=(0,0,255))
        img = img.crop(points).resize((100,100))
        img.show()
    tree.write('new_' + os.path.split(xml_file)[1])
    