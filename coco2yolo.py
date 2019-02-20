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
#from label_dog import DogCOCOLabelNames as COCOLabelNames
from label_chihuahua import ChihuahuaCOCOLabelNames as COCOLabelNames
resume=False

def main():
    #####
    # input: follow the procedure described on https://pjreddie.com/darknet/yolo/
    COCO_images_dir = '/data/huge/COCO/coco/images'
    COCO_labels_dir = '/data/huge/COCO/yolo/labels'
    COCOLabelNames.init('/data/work/dog/00input-chihuahua/chihuahua.txt', '/opt/darknet/data/coco.names')
    assert 'dog' in COCOLabelNames.label_names()
    assert 0 <= COCOLabelNames.label_index_src('dog')
    assert 0 == COCOLabelNames.label_index_dst('dog')
    assert 1 == COCOLabelNames.label_index_dst('chihuahua')
    splits = ( 'train2014', 'val2014', )

    #####
    # output
    OUT_dir = '/data/work/dog/00input-chihuahua'
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
                        if isinstance(id_, (tuple, list)):
                            for i in id_:
                                lines.append('{} {}'.format(i, bbox))
                        elif 0 <= id_:
                            lines.append('{} {}'.format(id_, bbox))
                        else:
                            use_for_negative = True
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
                    print('write {}'.format(bbox_out))
                    with open(bbox_out, 'w') as fbbox_out:
                        for line in lines:
                            fbbox_out.write(line)
                    flist.write('{}\n'.format(img))
                else:
                    print('skip {}'.format(os.path.basename(bbox_in)))

if __name__=='__main__':
    main()

# end of file
