#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

import os
import pathlib
import math
from xml.etree import ElementTree as ET
use_mapping=False
if use_mapping:
    #from label_dog import DogVOCLabelNames as VOCLabelNames
    from label_chihuahua import ChihuahuaVOCLabelNames as VOCLabelNames
else:
    from label_default import VOCLabelNames
resume=False

_min_x_w = 9999999999
_min_y_h = 9999999999
_max_x_w = 0
_max_y_h = 0
_max_w = 0
_max_h = 0
_max_w_h = 0
_max_h_w = 0
_have_difficult = False

def xml2yolo(xml, yolo, yolo_wo_difficult):
    global _min_x_w, _min_y_h, _max_x_w, _max_y_h
    global _max_w, _max_h, _max_w_h, _max_h_w
    global _have_difficult
    assert yolo, yolo
    xml = ET.parse(xml)

    tree_size = xml.find('size')
    w = float(tree_size.find('width').text)
    h = float(tree_size.find('height').text)

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

    lines = []
    lines_wo_difficult = []
    use_for_negative = False
    prev_ignore = ''
    for tree_obj in xml.findall('object'):
        label = tree_obj.find('name').text
        try:
            label_index = VOCLabelNames.label_index(label)
        except:
            assert use_mapping
            if prev_ignore != label:
                prev_ignore = label
                #print('skip {},'.format(label), end='', flush=True)
                print('.', end='', flush=True)
            continue
        if not isinstance(label_index, (tuple, list)) and 0 > label_index:
            assert use_mapping
            use_for_negative = True
            continue

        is_difficult = (0 != int(tree_obj.find('difficult').text.strip()))

        tree_bbox = tree_obj.find('bndbox')
        xmin, ymin, xmax, ymax = tuple(map(lambda k: float(tree_bbox.find(k).text.strip()) - 1, ('xmin', 'ymin', 'xmax', 'ymax')))
        assert 0 <= math.floor(xmin) < w, '{}: xmin: {}, {}'.format(xml, xmin, w)
        assert 0 <= math.floor(ymin) < h, '{}: ymin: {}, {}'.format(xml, ymin, h)
        assert 0 <= math.ceil(xmax)  < w, '{}: xmax: {}, {}'.format(xml, xmax, w)
        assert 0 <= math.ceil(ymax)  < h, '{}: ymax: {}, {}'.format(xml, ymax, h)
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
        for li in label_index if isinstance(label_index, (tuple, list)) else ( label_index, ):
            line = '{} {:1.15f} {:1.15f} {:1.15f} {:1.15f}\n'.format(
                    li,
                    (xmin + xmax) / 2.0 / (w - 1),
                    (ymin + ymax) / 2.0 / (h - 1),
                    (xmax - xmin + 1) / w,
                    (ymax - ymin + 1) / h)
            lines.append(line)
            if is_difficult:
                _have_difficult = True
            else:
                lines_wo_difficult.append(line)

    if lines_wo_difficult and yolo_wo_difficult:
        os.makedirs(os.path.dirname(yolo_wo_difficult), exist_ok=True)
        with open(yolo_wo_difficult, 'w', newline='\n') as f_wo_difficult:
            for line in lines_wo_difficult:
                f_wo_difficult.write(line)

    is_used = False
    if use_for_negative or lines:
        is_used = True
        os.makedirs(os.path.dirname(yolo), exist_ok=True)
        with open(yolo, 'w', newline='\n') as f:
            for line in lines:
                f.write(line)
        print('{},'.format(yolo), end='', flush=True)

    return is_used

def main():
    global _min_x_w, _min_y_h, _max_x_w, _max_y_h
    global _max_w, _max_h, _max_w_h, _max_h_w
    global _have_difficult

    #####
    # input
    #   label file: https://github.com/pjreddie/darknet/blob/master/data/voc.names
    VOC_dir = '/data/huge/VOC/VOCdevkit/VOC2012'
    if use_mapping:
        #VOCLabelNames.init('/data/work/dog/00input-dog/dog.txt', '/opt/darknet/data/voc.names')
        VOCLabelNames.init('/data/work/dog/00input-chihuahua/chihuahua.txt', '/opt/darknet/data/voc.names')
        assert 0 == VOCLabelNames.label_index_dst('dog')
        assert 0 <= VOCLabelNames.label_index_src('dog')
        assert 0 > VOCLabelNames.label_index('cat')
    else:
        VOCLabelNames.init('/opt/darknet/data/voc.names')
        assert 0 <= VOCLabelNames.label_index('dog')
    files = {
        'train': os.path.join(VOC_dir, 'ImageSets/Main/train.txt'),
          'val': os.path.join(VOC_dir, 'ImageSets/Main/val.txt'),
    }

    #####
    # output
    if use_mapping:
        #OUT_dir = '/data/work/dog/00input-dog'
        OUT_dir = '/data/work/dog/00input-chihuahua'
    else:
        OUT_dir = '/data/huge/VOC/yolo/VOC2012'
    NAME='voc2012'

    for d in ( 'images', 'labels', 'labels-wo-difficult', 'lists' ):
        if not os.path.exists(os.path.join(OUT_dir, d)):
            os.makedirs(os.path.join(OUT_dir, d), exist_ok=True)
    if not os.path.exists(os.path.join(OUT_dir, 'images', NAME)):
        os.symlink(os.path.join(VOC_dir, 'JPEGImages'), os.path.join(OUT_dir, 'images', NAME))

    # create labels and lists
    with open(os.path.join(OUT_dir, 'lists', NAME + '.txt'), 'w') as fo:
        for split, file in files.items():
            with open(file, 'r') as fi:
                prev_yolo_dir = ''
                for line in fi:
                    id_ = line.split()
                    if not id_ or (2 <= len(id_) and int(id_[1]) < 1): continue
                    id_ = id_[0]
                    xml               = os.path.join(VOC_dir, 'Annotations',               id_ + '.xml')
                    yolo              = os.path.join(OUT_dir, 'labels',              NAME, id_ + '.txt')
                    yolo_wo_difficult = os.path.join(OUT_dir, 'labels-wo-difficult', NAME, id_ + '.txt')
                    img_stem = os.path.join('images', NAME, id_)
                    img = None
                    for ext in ( '.jpg', '.png', '.jpeg', '.JPEG' ):
                        if os.path.isfile(os.path.join(OUT_dir, img_stem + ext)):
                            img = img_stem + ext
                            break
                    if not img: raise FileNotFoundError(img_stem)
                    img = pathlib.PurePath(img).as_posix()
                    if xml2yolo(xml, yolo, yolo_wo_difficult):
                        fo.write('{}\n'.format(img))
                        cur_yolo_dir = os.path.dirname(yolo)
                        if prev_yolo_dir != cur_yolo_dir:
                            prev_yolo_dir = cur_yolo_dir
                            print('{} -> {}'.format(xml, yolo))
                    else:
                        assert use_mapping

    print('')
    print('max width : {}'.format(_max_w))
    print('max height: {}'.format(_max_h))
    print('max width  / height    : {:f}'.format(_max_w_h))
    print('max height / width     : {:f}'.format(_max_h_w))
    print('min xmin   / (width-1) : {:f}'.format(_min_x_w))
    print('min ymin   / (height-1): {:f}'.format(_min_y_h))
    print('max xmax   / (width-1) : {:f}'.format(_max_x_w))
    print('max ymax   / (height-1): {:f}'.format(_max_y_h))
    print('have difficult: {}'.format(_have_difficult))
    if not _have_difficult:
        print('remove redundant directory manually: {}'.format(os.path.join(OUT_dir, 'labels-wo-difficult')))

# =============================================================================
# max width : 500.0
# max height: 500.0
# max width  / height    : 4.545455
# max height / width     : 2.069307
# min xmin   / (width-1) : 0.000000
# min ymin   / (height-1): 0.000000
# max xmax   / (width-1) : 1.000000
# max ymax   / (height-1): 1.000000
# have difficult: True
# =============================================================================

if __name__=='__main__':
    main()

# end of file
