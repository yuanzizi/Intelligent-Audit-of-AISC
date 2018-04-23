# --coding:utf-8--
import cv2
import numpy as np
import matplotlib.image as mpimg


def get_sign_img(img_name):
    # img = cv2.imread(img_name)
    img = mpimg.imread(img_name)
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh2 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY)

    thresh2_ver = np.sum(thresh2, axis=0)/255
    thresh2_hor = np.sum(thresh2, axis=1)/255
    ver = len(thresh2_ver)
    hor = len(thresh2_hor)

    for i in range(ver-1):
        if thresh2_ver[i] != hor:
            ver_first = i
            break
        else:
            ver_first = 0

    for i in range(ver-1):
        if thresh2_ver[ver-i-1] != hor:
            ver_last = ver-i-1
            break
        else:
            ver_last = 0

    for i in range(hor-1):
        if thresh2_hor[i] != ver:
            hor_first = i
            break
        else:
            hor_first = 0

    for i in range(ver-1):
        if thresh2_hor[hor-i-1] != ver:
            hor_last = hor-i-1
            break
        else:
            hor_last = 0

    crop_thresh2 = thresh2[hor_first:hor_last, ver_first:ver_last]
    crop_thresh2 = cv2.resize(crop_thresh2, (192, 64))

    return crop_thresh2 / 255.0


def get_names_img(sign_name):
    img = mpimg.imread(sign_name)
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh2 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.resize(thresh2, (192, 64))

    return thresh2 / 255.0
