import glob
import json
import os
import shutil
import operator
import sys
import argparse

MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)


PATH =  os.path.join('..','0_Data','sign3','mAP')
PATH_PRED = os.path.join(PATH,'predicted')
PATH_GT = os.path.join(PATH,'ground-truth')

    
for path in [PATH_GT,PATH_PRED]:
    if not os.path.exists(path):
        print(path,'do not exist!')
        sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
# argparse receiving list of classes to be ignored
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
args = parser.parse_args()

# if there are no classes to ignore then replace None by empty list
if args.ignore is None:
  args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
  specific_iou_flagged = True

# if there are no images then no animation can be shown
img_path = os.path.join(PATH,'images')
if os.path.exists(img_path): 
  for dirpath, dirnames, files in os.walk(img_path):
    if not files:
      # no image files found
      args.no_animation = True
else:
  args.no_animation = True

# try to import OpenCV if the user didn't choose the option --no-animation
show_animation = False
if not args.no_animation:
  try:
    import cv2
    show_animation = True
  except ImportError:
    args.no_animation = True

# try to import Matplotlib if the user didn't choose the option --no-plot
draw_plot = False
if not args.no_plot:
  try:
    import matplotlib.pyplot as plt
    draw_plot = True
  except ImportError:
    args.no_plot = True

"""
 throw error and exit
"""
def error(msg):
  print(msg)
  sys.exit(0)

"""
 check if the number is a float between 0.0 and 1.0
"""
def is_float_between_0_and_1(value):
  try:
    val = float(value)
    if val > 0.0 and val < 1.0:
      return True
    else:
      return False
  except ValueError:
    return False

"""
 Calculate the AP given the recall and precision array
  1st) We compute a version of the measured precision/recall curve with
       precision monotonically decreasing
  2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
  """
  --- Official matlab code VOC2012---
  mrec=[0 ; rec ; 1];
  mpre=[0 ; prec ; 0];
  for i=numel(mpre)-1:-1:1
      mpre(i)=max(mpre(i),mpre(i+1));
  end
  i=find(mrec(2:end)~=mrec(1:end-1))+1;
  ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
  """
  rec.insert(0, 0.0) # insert 0.0 at begining of list
  rec.append(1.0) # insert 1.0 at end of list
  mrec = rec[:]
  prec.insert(0, 0.0) # insert 0.0 at begining of list
  prec.append(0.0) # insert 1.0 at end of list
  mpre = prec[:]
  # matlab indexes start in 1 but python in 0, so I have to do:
  #   range(start=(len(mpre) - 2), end=0, step=-1)
  # also the python function range excludes the end, resulting in:
  #   range(start=(len(mpre) - 2), end=-1, step=-1)
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])
  # matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
  ind = []
  for ind_1 in range(1, len(mrec)):
    if mrec[ind_1] != mrec[ind_1-1]:
      ind.append(ind_1) # if it was matlab would be ind_1 + 1
  # matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
  ap = 0.0
  for i in ind:
    ap += ((mrec[i]-mrec[i-1])*mpre[i])
  return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
  # open txt file lines to a list
  with open(path) as f:
    content = f.readlines()
  # remove whitespace characters like `\n` at the end of each line
  content = [x.strip() for x in content]
  return content

"""
 Draws text in image
"""
def draw_text_in_image(img, text, pos, color, line_width):
  font = cv2.FONT_HERSHEY_PLAIN
  fontScale = 1
  lineType = 1
  bottomLeftCornerOfText = pos
  cv2.putText(img, text,
      bottomLeftCornerOfText,
      font,
      fontScale,
      color,
      lineType)
  text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
  return img, (line_width + text_width)


"""
 Draw plot using Matplotlib
"""
def draw_plot_func(dictionary, n_classes, window_title, plot_title, y_label, output_path, to_show):
    # sort the dictionary by decreasing value (reverse=True), into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    plt.bar(range(n_classes), sorted_values, align='center')
    plt.xticks(range(n_classes), sorted_keys, rotation='vertical')
    # set window title
    fig = plt.gcf() # gcf - get current figure
    fig.canvas.set_window_title(window_title)
    # set plot title
    plt.title(plot_title)
    # set axis titles
    # plt.xlabel('classes')
    plt.ylabel(y_label)
    # adjust size of window
    fig.tight_layout()
    if to_show:
      plt.show()
    # save the plot
    fig.savefig(output_path)
    # clear the plot
    plt.clf()

"""
 Create a "tmp_files/" and "results/" directory
"""
tmp_files_path = os.path.join(PATH, "tmp_files")
if not os.path.exists(tmp_files_path): # if it doesn't exist already
  os.makedirs(tmp_files_path)
results_files_path = os.path.join(PATH,"results")
if os.path.exists(results_files_path): # if it exist already
  # reset the results directory
  shutil.rmtree(results_files_path)

os.makedirs(results_files_path)
if draw_plot:
  os.makedirs(results_files_path + "/classes")
if show_animation:
  os.makedirs(results_files_path + "/images")

"""
 Ground-Truth
   Load each of the ground-truth files into a temporary ".json" file.
   Create a list of all the class names present in the ground-truth (unique_classes).
"""
# get a list with the ground-truth files
ground_truth_files_list = glob.glob(os.path.join(PATH_GT,'*.txt'))
ground_truth_files_list.sort()
gt_counter_per_class = {}

unique_classes = set([])
# dictionary with counter per class

for txt_file in ground_truth_files_list:
  #print(txt_file)
  file_id = txt_file.split(".txt",1)[0]
  file_id = os.path.basename(os.path.normpath(file_id))
  # check if there is a correspondent predicted objects file
  if not os.path.exists(os.path.join(PATH_PRED,file_id + ".txt")):
    error("Error. File not found: predicted/" +  file_id + ".txt")
  lines_list = file_lines_to_list(txt_file)
  # create ground-truth dictionary
  bounding_boxes = []
  for line in lines_list:
    try:
      class_name, left, top, right, bottom = line.split()
    except ValueError:
      print("Error: File " + txt_file + " in the wrong format.")
      print(" Expected: <class_name> <left> <top> <right> <bottom>")
      print(" Received: " + line)
      sys.exit(0)
    # check if class is in the ignore list, if yes skip
    if class_name in args.ignore:
      continue
    bbox = left + " " + top + " " + right + " " +bottom
    bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
    # count that object
    if class_name in gt_counter_per_class:
      gt_counter_per_class[class_name] += 1
    else:
      # if class didn't exist yet
      gt_counter_per_class[class_name] = 1
      unique_classes.add(class_name)
  # dump bounding_boxes into a ".json" file
  with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
    json.dump(bounding_boxes, outfile)

# let's sort the classes alphabetically
unique_classes = sorted(unique_classes)
n_classes = len(unique_classes)
#print(unique_classes)
#print(gt_counter_per_class)

"""
 Check format of the flag --set-class-iou (if used)
  e.g. check if class exists
"""
if specific_iou_flagged:
  n_args = len(args.set_class_iou)
  error_msg = \
    '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
  if n_args % 2 != 0:
    error('Error, missing arguments. Flag usage:' + error_msg)
  # [class_1] [IoU_1] [class_2] [IoU_2]
  # specific_iou_classes = ['class_1', 'class_2']
  specific_iou_classes = args.set_class_iou[::2] # even
  # iou_list = ['IoU_1', 'IoU_2']
  iou_list = args.set_class_iou[1::2] # odd
  if len(specific_iou_classes) != len(iou_list):
    error('Error, missing arguments. Flag usage:' + error_msg)
  for tmp_class in specific_iou_classes:
    if tmp_class not in unique_classes:
          error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
  for num in iou_list:
    if not is_float_between_0_and_1(num):
      error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

"""
 Plot the total number of occurences of each class in the ground-truth
"""
if draw_plot:
  window_title = "Ground-Truth Info"
  plot_title = "Total of ground-truth files = " + str(len(ground_truth_files_list))
  y_label = "Number of objects per class"
  output_path = results_files_path + "/Ground-Truth Info.jpg"
  to_show = False
  draw_plot_func(gt_counter_per_class, n_classes, window_title, plot_title, y_label, output_path, to_show)

"""
 Write number of ground-truth objects per class to results.txt
"""
with open(results_files_path + "/results.txt", 'w') as results_file:
  results_file.write("# Number of ground-truth objects per class\n")
  for class_name in sorted(gt_counter_per_class):
    results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

"""
 Predicted
   Load each of the predicted files into a temporary ".json" file.
"""
# get a list with the predicted files
predicted_files_list = glob.glob(os.path.join(PATH_PRED,'*.txt'))
predicted_files_list.sort()
pred_counter_per_class = {}

for class_name in unique_classes:
  bounding_boxes = []
  pred_counter_per_class[class_name] = 0
  for txt_file in predicted_files_list:
    #print txt_file
    lines = file_lines_to_list(txt_file)
    for line in lines:
      line_class_name, confidence, left, top, right, bottom = line.split()
      if line_class_name == class_name:
        #print("match")
        file_id = txt_file.split(".txt",1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        bbox = left + " " + top + " " + right + " " +bottom
        bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
        #print(bounding_boxes)
        # count that object
        if class_name in pred_counter_per_class:
          pred_counter_per_class[class_name] += 1
        else:
          # if class didn't exist yet
          pred_counter_per_class[class_name] = 1
  # sort predictions by decreasing confidence
  bounding_boxes.sort(key=lambda x:x['confidence'], reverse=True)
  with open(tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
    json.dump(bounding_boxes, outfile)

"""
 Plot the total number of occurences of each class in the "predicted" folder
"""
if draw_plot:
  window_title = "Predicted Objects Info"
  plot_title = "Total of Predicted Objects files = " + str(len(predicted_files_list))
  y_label = "Number of objects per class"
  output_path = results_files_path + "/Predicted Objects Info.jpg"
  to_show = False
  draw_plot_func(pred_counter_per_class, n_classes, window_title, plot_title, y_label, output_path, to_show)

"""
 Write number of predicted objects per class to results.txt
"""
with open(results_files_path + "/results.txt", 'a') as results_file:
  results_file.write("\n# Number of predicted objects per class\n")
  for class_name in sorted(pred_counter_per_class):
    results_file.write(class_name + ": " + str(pred_counter_per_class[class_name]) + "\n")

"""
 Calculate the AP for each class
"""
sum_AP = 0.0
ap_dictionary = {}
# open file to store the results
with open(results_files_path + "/results.txt", 'a') as results_file:
  results_file.write("\n# AP and precision/recall per class\n")
  for class_index, class_name in enumerate(unique_classes):
    """
     Load predictions of that class
    """
    predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
    predictions_data = json.load(open(predictions_file))

    """
     Assign predictions to ground truth objects
    """
    nd = len(predictions_data)
    tp = [0] * nd # creates an array of zeros of size nd
    fp = [0] * nd
    for idx, prediction in enumerate(predictions_data):
      file_id = prediction["file_id"]
      if show_animation:
        # find ground truth image
        ground_truth_img = glob.glob1(img_path, file_id + ".*")
        #tifCounter = len(glob.glob1(myPath,"*.tif"))
        if len(ground_truth_img) == 0:
          error("Error. Image not found with id: " + file_id)
        elif len(ground_truth_img) > 1:
          error("Error. Multiple image with id: " + file_id)
        else: # found image
          #print(img_path + "/" + ground_truth_img[0])
          # Load image
          img = cv2.imread(img_path + "/" + ground_truth_img[0])
          # Add bottom border to image
          bottom_border = 60
          BLACK = [0, 0, 0]
          img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
      # assign prediction to ground truth object if any
      #   open ground-truth with that file_id
      gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
      ground_truth_data = json.load(open(gt_file))
      ovmax = -1
      gt_match = -1
      # load prediction bounding-box
      bb = [ float(x) for x in prediction["bbox"].split() ]
      for obj in ground_truth_data:
        # look for a class_name match
        if obj["class_name"] == class_name:
          bbgt = [ float(x) for x in obj["bbox"].split() ]
          bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
          iw = bi[2] - bi[0] + 1
          ih = bi[3] - bi[1] + 1
          if iw > 0 and ih > 0:
            # compute overlap (IoU) = area of intersection / area of union
            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                    + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
            ov = iw * ih / ua
            if ov > ovmax:
              ovmax = ov
              gt_match = obj

      # assign prediction as true positive or false positive
      if show_animation:
        status = "NO MATCH FOUND!" # status is only used in the animation
      # set minimum overlap
      min_overlap = MINOVERLAP
      if specific_iou_flagged:
        if class_name in specific_iou_classes:
          index = specific_iou_classes.index(class_name)
          min_overlap = float(iou_list[index])
      if ovmax >= min_overlap:
        if not bool(gt_match["used"]):
          # true positive
          tp[idx] = 1
          gt_match["used"] = True
          # update the ".json" file
          with open(gt_file, 'w') as f:
              f.write(json.dumps(ground_truth_data))
          if show_animation:
            status = "MATCH!"
        else:
          # false positive (multiple detection)
          fp[idx] = 1
          if show_animation:
            status = "REPEATED MATCH!"
      else:
        # false positive
        fp[idx] = 1
        if ovmax > 0:
          status = "INSUFFICIENT OVERLAP"

      """
       Draw image to show animation
      """
      if show_animation:
        height, widht = img.shape[:2]
        # colors (OpenCV works with BGR)
        white = (255,255,255)
        light_blue = (255,200,100)
        green = (0,255,0)
        light_red = (30,30,255)
        # 1st line
        margin = 10
        v_pos = int(height - margin - (bottom_border / 2))
        text = "Image: " + ground_truth_img[0] + " "
        img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
        text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
        img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
        if ovmax != -1:
          color = light_red
          if status == "INSUFFICIENT OVERLAP":
            text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100)
          else:
            text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
            color = green
          img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
        # 2nd line
        v_pos += int(bottom_border / 2)
        text = "Prediction confidence: {0:.2f}% ".format(float(prediction["confidence"])*100)
        img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
        color = light_red
        if status == "MATCH!":
          color = green
        text = "Result: " + status + " "
        img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

        if ovmax > 0: # if there is intersections between the bounding-boxes
          bbgt = [ float(x) for x in gt_match["bbox"].split() ]
          cv2.rectangle(img,(int(bbgt[0]),int(bbgt[1])),(int(bbgt[2]),int(bbgt[3])),light_blue,2)
        if status == "MATCH!":
          cv2.rectangle(img,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),green,2)
        else:
          cv2.rectangle(img,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),light_red,2)
        cv2.imshow("Animation", img)
        cv2.waitKey(20) # show image for 20 ms
        # save image to results
        output_img_path = results_files_path + "/images/" + class_name + "_prediction" + str(idx) + ".jpg"
        cv2.imwrite(output_img_path, img)

    #print(tp)
    # compute precision/recall
    cumsum = 0
    for idx, val in enumerate(fp):
      fp[idx] += cumsum
      cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
      tp[idx] += cumsum
      cumsum += val
    #print(tp)
    rec = tp[:]
    for idx, val in enumerate(tp):
      rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
    #print(rec)
    prec = tp[:]
    for idx, val in enumerate(tp):
      prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    #print(prec)

    ap, mrec, mprec = voc_ap(rec, prec)
    sum_AP += ap
    text = class_name + " AP = {0:.2f}%".format(ap*100)
    """
     Write to results.txt
    """
    rounded_prec = [ '%.2f' % elem for elem in prec ]
    rounded_rec = [ '%.2f' % elem for elem in rec ]
    results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
    if not args.quiet:
      print(text)
    ap_dictionary[class_name] = ap

    """
     Draw plot
    """
    if draw_plot:
      plt.plot(rec, prec, '-o')
      plt.fill_between(mrec, 0, mprec, alpha=0.2, edgecolor='r')
      # set window title
      fig = plt.gcf() # gcf - get current figure
      fig.canvas.set_window_title('AP ' + class_name)
      # set plot title
      plt.title('class: ' + text)
      #plt.suptitle('This is a somewhat long figure title', fontsize=16)
      # set axis titles
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      # optional - set axes
      axes = plt.gca() # gca - get current axes
      axes.set_xlim([0.0,1.0])
      axes.set_ylim([0.0,1.05]) # .05 to give some extra space
      # Alternative option -> wait for button to be pressed
      #while not plt.waitforbuttonpress(): pass # wait for key display
      # Alternative option -> normal display
      #plt.show()
      # save the plot
      fig.savefig(results_files_path + "/classes/" + class_name + ".jpg")
      plt.cla() # clear axes for next plot

  if show_animation:
    cv2.destroyAllWindows()

  results_file.write("\n# mAP of all classes\n")
  mAP = sum_AP / n_classes
  text = "mAP = {0:.2f}%".format(mAP*100)
  results_file.write(text + "\n")
  print(text)

# remove the tmp_files directory
shutil.rmtree(tmp_files_path)

"""
 Draw mAP plot (Show AP's of all classes in decreasing order)
"""
if draw_plot:
  window_title = "mAP"
  plot_title = "mAP = {0:.2f}%".format(mAP*100)
  y_label = "Average Precision"
  output_path = results_files_path + "/mAP.jpg"
  to_show = True
  draw_plot_func(ap_dictionary, n_classes, window_title, plot_title, y_label, output_path, to_show)
