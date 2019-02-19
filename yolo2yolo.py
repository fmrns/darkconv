#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

# convert validation results of yolo to annotations of yolo

import os
import glob
import pathlib
import math
import cv2
from dog_label import DogLabelNames

# darknet/src/utils.c
def find_replace(s, orig, targ):
    return s.replace(orig, targ, 1)

_min_x_w = 9999999999
_min_y_h = 9999999999
_max_x_w = 0
_max_y_h = 0
_max_w = 0
_max_h = 0
_max_w_h = 0
_max_h_w = 0

def write_label(img_dir, img_id, label, bboxes):
    global _min_x_w, _min_y_h, _max_x_w, _max_y_h
    global _max_w, _max_h, _max_w_h, _max_h_w
    img = None
    for f in glob.glob(os.path.join(img_dir, img_id + '.*')):
        if f.endswith('.txt'):
            continue
        if img:
            raise FileExistsError('{} and\n{}'.format(img, f))
        img = f
    if img:
        img = cv2.imread(f)
    if img is None:
        raise FileNotFoundError(os.path.join(img_dir, img_id))

    h, w = img.shape[:2]

    if _max_w < w:
        _max_w = w
        print('New max width: {}'.format(w))
    if _max_h < h:
        _max_h = h
        print('New max height: {}'.format(h))
    if h < w:
        w_h = w / h
        if _max_w_h < w_h:
            _max_w_h = w_h
            print('New width / height: {}'.format(w_h))
    else:
        h_w = h / w
        if _max_h_w < h_w:
            _max_h_w = h_w
            print('New height / width: {}'.format(h_w))

    with open(find_replace(os.path.join(img_dir, img_id + '.txt.' + label), 'images', 'labels'), 'w') as flabel:
        for bbox in bboxes:
            xmin = max(0, bbox[0])
            ymin = max(0, bbox[1])
            xmax = min(w - 1, bbox[2])
            ymax = min(h - 1, bbox[3])

            if _min_x_w > xmin / w:
                _min_x_w = xmin / (w - 1)
                print('New min xmin / (width-1): {}'.format(_min_x_w))
            if _min_y_h > ymin / h:
                _min_y_h = ymin / (h - 1)
                print('New min ymin / (height-1): {}'.format(_min_y_h))
            if _max_x_w < xmax / (w - 1):
                _max_x_w = xmax / (w - 1)
                print('New max xmax / (width-1): {}'.format(_max_x_w))
            if _max_y_h < ymax / (h - 1):
                _max_y_h = ymax / (h - 1)
                print('New max ymax / (height-1): {}'.format(_max_y_h))

            # 0, 1:
            #  width=2
            #  center = (0 + 1) / 2 / (2-1) = 0.5
            #  w = (1 - 0 + 1) / 2 = 1
            # 0, 2:
            #  width=3
            #  center = (0 + 2) / 2 / (3-1) = 0.5
            #  w = (2 - 0 + 1) / 3 = 1
            # 0, 3:
            #  width = 4
            #  center = (0 + 3) / 2 / (4-1) = 0.5
            #  w = (3 - 0 + 1) / 4 = 1
            flabel.write('{:1.15f} {:1.15f} {:1.15f} {:1.15f}\n'.format(
                (xmin + xmax) / 2.0 / (w - 1),
                (ymin + ymax) / 2.0 / (h - 1),
                (xmax - xmin + 1) / w,
                (ymax - ymin + 1) / h))

def main():
    global _min_x_w, _min_y_h, _max_x_w, _max_y_h
    global _max_w, _max_h, _max_w_h, _max_h_w

    eps = 10e-5
    prec_threshold = 0
    images_dir = '/data/work/dog/00test-dog/images'
    labels_dir = find_replace(images_dir, 'images', 'labels')
    results_dir = 'results'
    predefined_labels = '/data/work/dog/00input-dog/dog.txt'
    DogLabelNames.init(predefined_labels)
    labelImg = []
    for label in DogLabelNames.label_names():
        for pred in glob.glob(pathlib.PurePath(results_dir).as_posix() + '/**/comp4_det_test_' + label + '.txt'):
            bboxes = []
            bboxes_id = ''
            rel_dir = os.path.dirname(os.path.relpath(pred, start=results_dir))
            img_dir = os.path.join(images_dir, rel_dir)
            lbl_dir = os.path.join(labels_dir, rel_dir)
            os.makedirs(lbl_dir, exist_ok=True)
            cls_file = os.path.join(lbl_dir, 'classes.txt')
            DogLabelNames.save(cls_file)
            labelImg.append('python3 /opt/labelImg/labelImg.py {} {} {}'.format(
                    pathlib.PurePath(img_dir).as_posix(),
                    pathlib.PurePath(cls_file).as_posix(),
                    pathlib.PurePath(lbl_dir).as_posix()))
            labelImg.append('diff -auw {} {}'.format(
                    pathlib.PurePath(predefined_labels).as_posix(),
                    pathlib.PurePath(cls_file).as_posix()))
            print('Reading: {}'.format(pred))
            with open(pred, 'r') as fpred:
                for line in fpred:
                    id_, *flts = line.split()
                    if bboxes and id_ != bboxes_id:
                        print(',{}'.format(bboxes_id), end='', flush=True)
                        write_label(img_dir, bboxes_id, label, bboxes)
                        bboxes = []
                    bboxes_id = id_
                    prec, xmin, ymin, xmax, ymax = map(lambda x: float(x) - 1, flts)
                    xmin_r, ymin_r, xmax_r, ymax_r = map(round, ( xmin, ymin, xmax, ymax ))
                    dlt = xmin_r - xmin
                    xmin = math.floor(xmin) if dlt > 0 and dlt > eps else xmin_r
                    dlt = ymin_r - ymin
                    ymin = math.floor(ymin) if dlt > 0 and dlt > eps else ymin_r
                    dlt = xmax_r - xmax
                    xmax = math.ceil(xmax) if dlt < 0 and -dlt > eps else xmax_r
                    dlt = ymax_r - ymax
                    ymax = math.ceil(ymax) if dlt < 0 and -dlt > eps else ymax_r
                    if 1 > len(bboxes) or prec >= prec_threshold:
                        bboxes.append(( xmin, ymin, xmax, ymax ))
                if bboxes:
                    print(',{}'.format(bboxes_id), end='')
                    write_label(img_dir, bboxes_id, label, bboxes)

    for file in glob.glob(pathlib.PurePath(labels_dir).as_posix() + '/**/*.txt.*', recursive=True):
        lbl_dir = os.path.dirname(file)
        lbl_file = os.path.join(lbl_dir, pathlib.PurePath(file).stem)
        if not lbl_file.endswith('.txt'):
            raise FileExistsError('Possible cause of some mistakes: {}'.format(lbl_file))
        if os.path.isfile(lbl_file) and os.path.getmtime(lbl_file) - os.path.getmtime(file) > 0:
            continue
        with open(lbl_file, 'w', newline='\n') as flbl:
            for filex in glob.glob(pathlib.PurePath(lbl_file).as_posix() + '.*'):
                ext = pathlib.PurePath(filex).suffix[1:]
                if ext not in DogLabelNames.label_names():
                    raise ValueError('Unknown label: {}, {}'.format(ext, lbl_file))
                lbl =  DogLabelNames.label_index(ext)
                assert 0 <= lbl, '{}: {}'.format(ext, lbl)
                with open(filex, 'r', newline='\n') as f:
                    for line in f:
                        flbl.write('{} {}'.format(lbl, line))
        print(',{}'.format(lbl_file), end='', flush=True)

    print('')
    print('max width : {}'.format(_max_w))
    print('max height: {}'.format(_max_h))
    print('max width  / height    : {:f}'.format(_max_w_h))
    print('max height / width     : {:f}'.format(_max_h_w))
    print('min xmin   / (width-1) : {:f}'.format(_min_x_w))
    print('min ymin   / (height-1): {:f}'.format(_min_y_h))
    print('max xmax   / (width-1) : {:f}'.format(_max_x_w))
    print('max ymax   / (height-1): {:f}'.format(_max_y_h))
    if labelImg:
        print('To edit:')
        for line in labelImg:
            print(line)

if __name__ == '__main__':
    main()

# end of file
