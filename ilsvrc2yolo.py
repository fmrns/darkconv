#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

import os
import pathlib
import contextlib
from xml.etree import ElementTree as ET
from bbox import BBox
use_mapping=True
if not use_mapping:
    from label_default import ILSVRCLabelNames
    ILSVRCLabelNames.init('/data/huge/ILSVRC/LOC_synset_mapping.txt')
    assert 0 <= ILSVRCLabelNames.label_index('n02085620')
    assert 0 <= ILSVRCLabelNames.label_index('n02085782')
    assert 0 <= ILSVRCLabelNames.label_index('n02088364')
    OUT_dir = '/data/huge/ILSVRC/yolo'
elif True:
    from label_chihuahua import ChihuahuaILSVRCLabelNames as ILSVRCLabelNames
    ILSVRCLabelNames.init('/data/work/dog/00input-chihuahua/chihuahua.txt', '/data/huge/ILSVRC/LOC_synset_mapping.txt')
    assert set((1, 0)) == set(ILSVRCLabelNames.label_index('n02085620'))
    assert 0 == ILSVRCLabelNames.label_index('n02085782')
    assert 0 == ILSVRCLabelNames.label_index('n02088364')
    OUT_dir = '/data/work/dog/00input-chihuahua'
else:
    from label_dog import DogILSVRCLabelNames as ILSVRCLabelNames
    ILSVRCLabelNames.init('/data/work/dog/00input-dog/dog.txt', '/data/huge/ILSVRC/LOC_synset_mapping.txt')
    assert 0 == ILSVRCLabelNames.label_index('n02085620')
    assert 0 == ILSVRCLabelNames.label_index('n02085782')
    assert 0 == ILSVRCLabelNames.label_index('n02088364')
    OUT_dir = '/data/work/dog/00input-dog'
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
    global resume
    global _min_x_w, _min_y_h, _max_x_w, _max_y_h
    global _max_w, _max_h, _max_w_h, _max_h_w
    global _have_difficult
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
    prev_ignore = ''
    use_for_negative = False
    for tree_obj in xml.findall('object'):
        label = tree_obj.find('name').text
        try:
            label_index = ILSVRCLabelNames.label_index(label)
        except:
            assert use_mapping
            if prev_ignore != label:
                prev_ignore = label
                #print(',Ignoring {}'.format(label), end='', flush=True)
                print('.', end='', flush=True)
            continue
        if not isinstance(label_index, (tuple, list)) and 0 > label_index:
            assert use_mapping
            use_for_negative = True
            continue

        is_difficult = (0 != int(tree_obj.find('difficult').text.strip()))

        tree_bbox = tree_obj.find('bndbox')
        bbox = BBox(hw=(h, w), type_=BBox.ILSVRC, bbox=tuple(map(lambda k: float(tree_bbox.find(k).text.strip()), ('xmin', 'ymin', 'xmax', 'ymax'))))
        xmin, ymin, xmax, ymax = bbox.get(type_=BBox.OPEN_IMAGES)
        if _min_x_w > xmin:
            _min_x_w = xmin
            print('New min xmin / (width-1): {}'.format(_min_x_w))
        if _min_y_h > ymin:
            _min_y_h = ymin
            print('New min ymin / (height-1): {}'.format(_min_y_h))
        if _max_x_w < xmax:
            _max_x_w = xmax
            print('New max xmax / (width-1): {}'.format(_max_x_w))
        if _max_y_h < ymax:
            _max_y_h = ymax
            print('New max ymax / (height-1): {}'.format(_max_y_h))
        for li in label_index if isinstance(label_index, (tuple, list)) else ( label_index, ):
            line = '{} {:1.15f} {:1.15f} {:1.15f} {:1.15f}\n'.format(li, *bbox.get(type_=BBox.YOLO))
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
        if os.path.exists(yolo):
            if resume: return is_used
            raise FileExistsError(yolo)
        os.makedirs(os.path.dirname(yolo), exist_ok=True)
        with open(yolo, 'w', newline='\n') as f:
            for line in lines:
                f.write(line)

    return is_used

def main():
    global _min_x_w, _min_y_h, _max_x_w, _max_y_h
    global _max_w, _max_h, _max_w_h, _max_h_w
    global _have_difficult
    global OUT_dir

    #####
    # input
    ILSVRC_dir = '/data/huge/ILSVRC'
    files = {
        'train': os.path.join(ILSVRC_dir, 'ImageSets/CLS-LOC/train_loc.txt'),
          'val': os.path.join(ILSVRC_dir, 'ImageSets/CLS-LOC/val.txt'),
         'test': os.path.join(ILSVRC_dir, 'ImageSets/CLS-LOC/test.txt'),
    }

    #####
    # output
    NAME='ilsvrc'

    for d in ( 'images', 'labels', 'labels-wo-difficult', 'lists' ):
        if not os.path.exists(os.path.join(OUT_dir, d)):
            os.makedirs(os.path.join(OUT_dir, d), exist_ok=True)
    if not os.path.exists(os.path.join(OUT_dir, 'images', NAME)):
        os.symlink(os.path.join(ILSVRC_dir, 'Data/CLS-LOC'), os.path.join(OUT_dir, 'images', NAME))

    # create labels and lists
    with open(os.path.join(OUT_dir, 'lists', NAME + '.txt'), 'w') if use_mapping else contextlib.nullcontext() as fom:
        for split, file in files.items():
            with open(file, 'r') as fi, \
              contextlib.nullcontext() if use_mapping else open(os.path.join(OUT_dir, 'lists', split + '.txt'), 'w') as fos:
                fo = fom if use_mapping else fos
                prev_yolo_dir = ''
                for line in fi:
                    id_ = line.split()
                    if not id_: continue
                    id_ = id_[0]
                    xml               = os.path.join(ILSVRC_dir, 'Annotations/CLS-LOC',    split, id_ + '.xml')
                    yolo              = os.path.join(OUT_dir, 'labels',              NAME, split, id_ + '.txt')
                    yolo_wo_difficult = os.path.join(OUT_dir, 'labels-wo-difficult', NAME, split, id_ + '.txt')
                    img_stem = os.path.join('images', NAME, split, id_)
                    img = None
                    for ext in ('.JPEG', '.jpeg', '.jpg', '.png'):
                        if os.path.isfile(os.path.join(OUT_dir, img_stem + ext)):
                            img = img_stem + ext
                            break
                    if not img: raise FileNotFoundError(img_stem)
                    img = pathlib.PurePath(img).as_posix()
                    if os.path.exists(xml):
                        if xml2yolo(xml, yolo, yolo_wo_difficult):
                            fo.write('{}\n'.format(img))
                            cur_yolo_dir = os.path.dirname(yolo)
                            if prev_yolo_dir != cur_yolo_dir:
                                prev_yolo_dir = cur_yolo_dir
                                print('{} -> {}'.format(xml, yolo))
                        else:
                            assert use_mapping
                    elif not use_mapping:
                        assert 'test' == split
                        fo.write('{}\n'.format(img))

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
    if not use_mapping:
        print('to create trainval: cat lists/train.txt lists/val.txt >lists/trainval.txt')

# =============================================================================
# max width : 4992.0
# max height: 5935.0
# max width  / height    : 15.625000
# max height / width     : 11.147059
# min xmin   / (width-1) : 0.000000
# min ymin   / (height-1): 0.000000
# max xmax   / (width-1) : 1.000000
# max ymax   / (height-1): 1.000000
# have difficult: False
# remove redundant directory manually: labels-wo-difficult
# =============================================================================

if __name__=='__main__':
    main()

# end of file
