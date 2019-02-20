#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

import os
import pathlib
import glob
# mapping only. yolo -> yolo
use_mapping=True
from label_chihuahua import ChihuahuaGLabelNames as GLabelNames
resume=False

def main():
    #####
    # input
    YOLO_images_dir = '/data/huge/AI/G/images'
    YOLO_labels_dir = '/data/huge/AI/G/labels'
    YOLO_classes_txt = '/data/huge/AI/G/labels/classes.txt'
    GLabelNames.init('/data/work/dog/00input-chihuahua/chihuahua.txt', YOLO_classes_txt)
    assert 'G' in GLabelNames.label_names()
    assert 0 == GLabelNames.label_index_src('G')
    assert 2 == GLabelNames.label_index_dst('G')
    assert 1 == GLabelNames.label_index_dst('chihuahua')
    assert 0 == GLabelNames.label_index_dst('dog')
    assert set(( 2, 1, 0 )) == set(GLabelNames.label_index('G'))

    #####
    # output
    OUT_dir = '/data/work/dog/00input-chihuahua'
    NAME='G'

    for d in ( 'images', 'labels', 'lists' ):
        if not os.path.exists(os.path.join(OUT_dir, d)):
            os.makedirs(os.path.join(OUT_dir, d), exist_ok=True)
    if not os.path.exists(os.path.join(OUT_dir, 'images', NAME)):
        os.symlink(YOLO_images_dir, os.path.join(OUT_dir, 'images', NAME))
    os.makedirs(os.path.join(OUT_dir, 'labels', NAME), exist_ok=True)

    # create labels and lists
    with open(os.path.join(OUT_dir, 'lists', NAME + '.txt'), 'w') as flist:
        for bbox_in in glob.glob(pathlib.PurePath(os.path.join(YOLO_labels_dir, '*.txt')).as_posix(), recursive=False):
            if os.path.samefile(bbox_in, YOLO_classes_txt): continue
            lines = []
            use_for_negative = False
            with open(bbox_in, 'r', newline='\n') as fbbox_in:
                for line in fbbox_in:
                    try:
                        id_, bbox = line.split(maxsplit=1)
                    except ValueError:
                        print('{}: {}'.format(bbox_in, line))
                        raise
                    try:
                        id_ = GLabelNames.label_index(int(id_))
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
                rel = os.path.relpath(bbox_in, start=YOLO_labels_dir)
                img_stem = os.path.join('images', NAME, rel[:-4])
                img = None
                for ext in ('.jpg', '.png', '.jpeg', ):
                    for ex in ( ext, ext.upper() ):
                        if os.path.isfile(os.path.join(OUT_dir, img_stem + ex)):
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
