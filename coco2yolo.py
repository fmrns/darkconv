#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

# following the procedure described on https://pjreddie.com/darknet/yolo/ is the simplest.
# this is for mapping to my own dataset after that.

import os
import pathlib
import glob
use_mapping=True
from label_dog import DogCOCOLabelNames as COCOLabelNames
resume=False


def write_lines(flist, OUT_dir, NAME, split, image_id, lines):
    img_stem = os.path.join('images', NAME, split, image_id)
    img = None
    for ext in ( '.jpg', '.png', '.JPEG' ):
        if os.path.isfile(os.path.join(OUT_dir, img_stem + ext)):
            img = img_stem + ext
            break
    if not img: raise FileNotFoundError(img_stem)
    img = pathlib.PurePath(img).as_posix()
    yolo = os.path.join(OUT_dir, 'labels', NAME, split, image_id + '.txt')
    flist.write('{}\n'.format(img))
    if os.path.exists(yolo):
        if resume: return
        raise FileExistsError(yolo)
    with open(yolo, 'w', newline='\n') as f:
        for line in lines:
            f.write(line)
    print('Written: {}'.format(yolo))

if __name__=='__main__':
    #####
    # input: follow the procedure described on https://pjreddie.com/darknet/yolo/
    COCO_images_dir = '/data/huge/COCO/coco/images'
    COCO_labels_dir = '/data/huge/COCO/yolo/labels'
    COCOLabelNames.init('/data/work/dog/00input-dog/dog.txt', '/opt/darknet/data/coco.names')
    assert 0 == COCOLabelNames.label_index('dog')
    assert 'dog' in COCOLabelNames.label_names()
    splits = ( 'train2014', 'val2014', )

    #####
    # output
    OUT_dir = '/data/work/dog/00input-dog'
    NAME='coco'

    for d in ( 'images', 'labels', 'lists' ):
        if not os.path.exists(os.path.join(OUT_dir, d)):
            os.makedirs(os.path.join(OUT_dir, d), exist_ok=True)
    if not os.path.exists(os.path.join(OUT_dir, 'images', NAME)):
        os.symlink(COCO_images_dir, os.path.join(OUT_dir, 'images', NAME))
    for split in splits:
        os.makedirs(os.path.join(OUT_dir, 'labels', NAME, split), exist_ok=True)

    # create labels and lists
    with open(os.path.join(OUT_dir, 'lists', NAME + '.txt'), 'w') as flist:
        for split in splits:
            for bbox_in in glob.glob(pathlib.PurePath(os.path.join(COCO_labels_dir, split, '**/*.txt')).as_posix(), recursive=True):
                lines = []
                use_for_negative = False
                with open(bbox_in, 'r', newline='\n') as fbbox_in:
                    for line in fbbox_in:
                        id_, bbox = line.split(maxsplit=1)
                        try:
                            id_ = COCOLabelNames.label_index(int(id_))
                        except ValueError:
                            continue
                        if 0 > id_:
                            use_for_negative = True
                        else:
                            lines.append('{} {}'.format(id_, bbox))
                if use_for_negative or lines:
                    rel = os.path.relpath(bbox_in, start=COCO_labels_dir)
                    img_stem = os.path.join('images', NAME, rel[:-4])
                    img = None
                    for ext in ('.jpg', '.png', '.jpeg', '.JPEG' ):
                        if os.path.isfile(os.path.join(OUT_dir, img_stem + ext)):
                            img = img_stem + ext
                            break
                    if not img: raise FileNotFoundError(img_stem)
                    img = pathlib.PurePath(img).as_posix()
                    bbox_out = os.path.join(OUT_dir, 'labels', NAME, rel)
                    print('Writing {}...'.format(bbox_out))
                    with open(bbox_out, 'w') as fbbox_out:
                        for line in lines:
                            fbbox_out.write(line)
                    flist.write('{}\n'.format(img))
                else:
                    print('Skipping {}...'.format(bbox_in))

# end of file
