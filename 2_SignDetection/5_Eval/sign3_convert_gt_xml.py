import sys
import os
import glob
import xml.etree.ElementTree as ET

PATH = os.path.join('..','0_Data','sign3')
PATH_ANN = os.path.join(PATH,'Annotations')
PATH_SET = os.path.join(PATH,'ImageSets','main')
PATH_GT = os.path.join(PATH,'mAP','ground-truth')

# old files (xml format) will be moved to a "backup" folder
## create the backup dir if it doesn't exist already
for path in [PATH_ANN, PATH_SET]:
    if not os.path.exists(path):
      print('path does not exist!', path)

# create VOC format files
xml_list = glob.glob(os.path.join(PATH_ANN, '*.xml'))
xml_names = [os.path.split(i)[1] for i in xml_list]
if len(xml_list) == 0:
    print("Error: no .xml files found in ground-truth")
    sys.exit()
else:
    print('num of xml:', len(xml_list))

def read_text(name = 'test.txt'):
    
    names = []
    import codecs
    path = os.path.join(PATH_SET, name)
    if not os.path.exists(path):
        print('path does not exist!', path)
        return
    with codecs.open(path, encoding='utf-8', mode='rU') as f:
        for idx, line in enumerate(f):
            if idx > 99999:
                break
            names.append(line.strip())
            
    return names
    
test_names = read_text()
print('num fo test', len(test_names))

for name in test_names:
    
    if name + '.xml' not in xml_names:
        print(name,'not in xml_names')
        continue
    with open(os.path.join(PATH_GT, name+'.txt'), 'w') as new_f:
        xml_file = os.path.join(PATH_ANN,name+'.xml')
        root = ET.parse(xml_file).getroot()
        for obj in root.findall('object'):
          obj_name = obj.find('name').text
          bndbox = obj.find('bndbox')
          left = bndbox.find('xmin').text
          top = bndbox.find('ymin').text
          right = bndbox.find('xmax').text
          bottom = bndbox.find('ymax').text
          new_f.write(obj_name + " " + left + " " + top + " " + right + " " + bottom + '\n')



'''

for tmp_file in xml_list:
  #print(tmp_file)
  # 1. create new file (VOC format)
  with open(tmp_file.replace(".xml", ".txt"), "a") as new_f:
    root = ET.parse(tmp_file).getroot()
    for obj in root.findall('object'):
      obj_name = obj.find('name').text
      bndbox = obj.find('bndbox')
      left = bndbox.find('xmin').text
      top = bndbox.find('ymin').text
      right = bndbox.find('xmax').text
      bottom = bndbox.find('ymax').text
      new_f.write(obj_name + " " + left + " " + top + " " + right + " " + bottom + '\n')
  # 2. move old file (xml format) to backup
  os.rename(tmp_file, "backup/" + tmp_file)
  
'''