#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

import os
import pathlib
import contextlib
import pandas as pd
import csv
from bbox import BBox
use_mapping=False
if not use_mapping:
    from label_default import OpenImagesLabelNames
    OpenImagesLabelNames.init('/data/huge/OpenImages/labels/class-descriptions-boxable.csv')
    assert 0 <= OpenImagesLabelNames.label_index('/m/0bt9lr')
    OUT_dir = '/data/huge/OpenImages/yolo'
elif True:
    from label_chihuahua import ChihuahuaOpenImagesLabelNames as OpenImagesLabelNames
    OpenImagesLabelNames.init('/data/work/dog/00input-chihuahua/chihuahua.txt', '/data/huge/OpenImages/labels/class-descriptions-boxable.csv')
    assert '/m/0bt9lr' in OpenImagesLabelNames.label_names()
    assert 0 <= OpenImagesLabelNames.label_index_src('/m/0bt9lr')
    assert 0 == OpenImagesLabelNames.label_index_dst('dog')
    OUT_dir = '/data/work/dog/00input-chihuahua'
else:
    from label_dog import DogOpenImagesLabelNames as OpenImagesLabelNames
    OpenImagesLabelNames.init('/data/work/dog/00input-dog/dog.txt', '/data/huge/OpenImages/labels/class-descriptions-boxable.csv')
    assert '/m/0bt9lr' in OpenImagesLabelNames.label_names()
    assert 0 == OpenImagesLabelNames.label_index('/m/0bt9lr')
    assert 0 <= OpenImagesLabelNames.label_index_src('/m/0bt9lr')
    assert 0 == OpenImagesLabelNames.label_index_dst('dog')
    OUT_dir = '/data/work/dog/00input-dog'
resume=False

def write_lines(flist, OUT_dir, NAME, split, image_id, lines):
    global use_mapping, resume

    img_stem = os.path.join('images', NAME, split, image_id)
    img = None
    for ext in ( '.jpg', '.png', '.JPEG' ):
        if os.path.isfile(os.path.join(OUT_dir, img_stem + ext)):
            img = img_stem + ext
            break
    if not img: raise FileNotFoundError(img_stem)
    img = pathlib.PurePath(img).as_posix()
    flist.write('{}\n'.format(img))
    yolo = os.path.join(OUT_dir, 'labels', NAME, split, image_id + '.txt')
    if os.path.exists(yolo):
        if resume: return
        raise FileExistsError(yolo)
    with open(yolo, 'w', newline='\n') as f:
        for line in lines:
            f.write(line)
    if use_mapping:
        print('Written: {}'.format(yolo))
    else:
        print('.', end='', flush=True)

_min_xmin = 9999999999
_min_ymin = 9999999999
_max_xmax = 0
_max_ymax = 0

def main():
    global _min_xmin, _min_ymin, _max_xmax, _max_ymax, OUT_dir

    #####
    # input
    OpenImages_dir = '/data/huge/OpenImages'
    splits = ( 'train', 'validation', 'test' )
    files = ( 'labels/{}-annotations-bbox.csv', 'labels/{}-annotations-human-imagelabels-boxable.csv' )

    #####
    # output
    NAME='openimages'

    for d in ( 'images', 'labels', 'lists' ):
        if not os.path.exists(os.path.join(OUT_dir, d)):
            os.makedirs(os.path.join(OUT_dir, d), exist_ok=True)
    if not os.path.exists(os.path.join(OUT_dir, 'images', NAME)):
        os.symlink(os.path.join(OpenImages_dir, 'images'), os.path.join(OUT_dir, 'images', NAME))
    ignore_image_ids = {}
    for split in splits:
        os.makedirs(os.path.join(OUT_dir, 'labels', NAME, split), exist_ok=True)
        if use_mapping:
            #ImageID,Source,LabelName,Confidence
            #000002b66c9c498e,verification,/m/014j1m,0
            conf = pd.read_csv(os.path.join(OpenImages_dir, files[1].format(split)))
            print('confidence sources: {}'.format(conf['Source'].unique()))
            ids = conf.loc[(conf['Confidence']==0) & (conf['LabelName'].isin(OpenImagesLabelNames.label_names())), 'ImageID'].values.reshape(-1)
            ignore_image_ids[split] = ids
            print('Ignoring ids({}):{}'.format(split, ids))

    # create labels and lists
    with open(os.path.join(OUT_dir, 'lists', NAME + '.txt'), 'w') if use_mapping else contextlib.nullcontext() as fom:
        for split in splits:
            prev_skip = ''
            prev_nonveri = ''
            file = os.path.join(OpenImages_dir, files[0].format(split))
            with open(file, 'r') as fi, \
              contextlib.nullcontext() if use_mapping else open(os.path.join(OUT_dir, 'lists', split + '.txt'), 'w') as fos:
                fo = fom if use_mapping else fos
                #ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
                #000026e7ee790996,freeform,/m/07j7r,1,0.071905,0.145346,0.206591,0.391306,0,1,1,0,0
                lines = []
                lines_id = ''
                use_for_negative = False
                for row in csv.DictReader(fi):
                    if (use_for_negative or lines) and lines_id != row['ImageID']:
                        write_lines(fo, OUT_dir, NAME, split, lines_id, lines)
                        lines = []
                        lines_id = ''
                        use_for_negative = False
                    try:
                        label_index = OpenImagesLabelNames.label_index(row['LabelName'])
                    except:
                        assert use_mapping
                        if prev_skip != row['ImageID'][:2]:
                            prev_skip = row['ImageID'][:2]
                            print('Skipping {}:{}...'.format(split, prev_skip))
                        continue
                    if not isinstance(label_index, (tuple, list)) and 0 > label_index:
                        assert use_mapping
                        use_for_negative = True
                        lines_id = row['ImageID']
                        continue
                    if ignore_image_ids and row['ImageID'] in ignore_image_ids[split]:
                        if prev_nonveri != row['ImageID'][:2]:
                            prev_nonveri = row['ImageID'][:2]
                            print('Not verified {}:{}...'.format(split, prev_nonveri))
                        continue

                    bbox = BBox(type_=BBox.OPEN_IMAGES, bbox=(float(row['XMin']), float(row['YMin']), float(row['XMax']), float(row['YMax'])))
                    xmin, ymin, xmax, ymax = bbox.get(type_=BBox.OPEN_IMAGES)
                    if  _min_xmin > xmin:
                        _min_xmin = xmin
                    if  _min_ymin > ymin:
                        _min_ymin = ymin
                    if  _max_xmax < xmax:
                        _max_xmax = xmax
                    if  _max_ymax < ymax:
                        _max_ymax = ymax
                    for li in label_index if isinstance(label_index, (tuple, list)) else ( label_index, ):
                        line = '{} {:1.7f} {:1.7f} {:1.7f} {:1.7f}\n'.format(li, *bbox.get(type_=BBox.YOLO))
                        lines.append(line)
                    assert '' == lines_id or row['ImageID'] == lines_id, '{}, {}'.format(row['ImageID'], lines_id)
                    lines_id = row['ImageID']

                if use_for_negative or lines:
                    write_lines(fo, OUT_dir, NAME, split, lines_id, lines)

    print('')
    print('min xmin: {:f}'.format(_min_xmin))
    print('min ymin: {:f}'.format(_min_ymin))
    print('max xmax: {:f}'.format(_max_xmax))
    print('max ymax: {:f}'.format(_max_ymax))

# =============================================================================
# =============================================================================

if __name__=='__main__':
    main()

# end of file
