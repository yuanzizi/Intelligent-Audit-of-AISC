# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from argparse import RawTextHelpFormatter
import fnmatch
import os
import cv2
import json
import random
import numpy as np
import shutil
from lang_aux import LangCharsGenerate
from lang_aux import FontCheck
from lang_aux import Font2Image

if __name__ == "__main__":

    description = '''
        make_ocr_dataset --output_dir /root/data/caffe_dataset \
            --font_dir /root/workspace/deep_ocr_fonts/chinese_fonts \
            --width 30 --height 30 --margin 4 --langs lower_eng
    '''

    parser = argparse.ArgumentParser(
        description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='E:/mobile_program/model/generate_names/single_word_generate', required=False,
                        help='write a caffe dir')
    parser.add_argument('--font_dir', dest='font_dir',
                        default='E:/mobile_program/model/character/font', required=False,
                        help='font dir to to produce images')
    parser.add_argument('--test_ratio', dest='test_ratio',
                        default=0.3, required=False,
                        help='test dataset size')
    parser.add_argument('--width', dest='width',
                        default=64, required=False,
                        help='width')
    parser.add_argument('--height', dest='height',
                        default=64, required=False,
                        help='height')
    parser.add_argument('--no_crop', dest='no_crop',
                        default=True, required=False,
                        help='', action='store_true')
    parser.add_argument('--margin', dest='margin',
                        default=4, required=False,
                        help='', )
    parser.add_argument('--langs', dest='langs',
                        default="chi_sim", required=False,
                        help='deep_ocr.langs.*, e.g. chi_sim, chi_tra, digits...')
    options = parser.parse_args()

    out_caffe_dir = os.path.expanduser(options.output_dir)
    font_dir = os.path.expanduser(options.font_dir)
    test_ratio = float(options.test_ratio)
    width = int(options.width)
    height = int(options.height)
    need_crop = not options.no_crop
    margin = int(options.margin)
    langs = options.langs

    image_dir_name = "images"

    images_dir = out_caffe_dir + '/' + image_dir_name
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    lang_chars_gen = LangCharsGenerate(langs)
    lang_chars = lang_chars_gen.do()
    font_check = FontCheck(lang_chars)

    y_to_tag = {}
    y_tag_json_file = os.path.join(out_caffe_dir, "y_tag.json")
    y_tag_text_file = os.path.join(out_caffe_dir, "y_tag.txt")
    path_train = os.path.join(out_caffe_dir, "train.txt")
    path_test = os.path.join(out_caffe_dir, "test.txt")

    verified_font_paths = []
    ## search for file fonts
    for font_name in os.listdir(font_dir):
        path_font_file = os.path.join(font_dir, font_name)
        if font_check.do(path_font_file):
            verified_font_paths.append(path_font_file)

    train_list = []
    test_list = []
    max_train_i = int(len(verified_font_paths) * (1.0 - test_ratio))

    font2image = Font2Image(width, height, need_crop, margin)

    for i, verified_font_path in enumerate(verified_font_paths):
        is_train = True
        if i >= max_train_i:
            is_train = False
        for j, char in enumerate(lang_chars):
            if j not in y_to_tag:
                y_to_tag[j] = char
            char_dir = os.path.join(images_dir, "%d" % j)
            if not os.path.isdir(char_dir):
                os.makedirs(char_dir)
            path_image = os.path.join(
                char_dir,
                "%d_%s.jpg" % (i, os.path.basename(verified_font_path)))
            relative_path_image = os.path.join(
                image_dir_name, "%d" % j,
                                "%d_%s.jpg" % (i, os.path.basename(verified_font_path))
            )
            font2image.do(verified_font_path, char, path_image)
            if is_train:
                train_list.append((relative_path_image, j))
            else:
                test_list.append((relative_path_image, j))

    h_y_tag_json_file = open(y_tag_json_file, "w+")
    json.dump(y_to_tag, h_y_tag_json_file)
    h_y_tag_json_file.close()

    h_y_tag_text_file = open(y_tag_text_file, "w+")
    for key in y_to_tag:
        h_y_tag_text_file.write("%d %s\n" % (key, y_to_tag[key].encode("utf-8")))
    h_y_tag_text_file.close()

    fout = open(path_train, "w+")
    for item in train_list:
        fout.write("%s %d\n" % (item[0], item[1]))
    fout.close()

    fout = open(path_test, "w+")
    for item in test_list:
        fout.write("%s %d\n" % (item[0], item[1]))
    fout.close()