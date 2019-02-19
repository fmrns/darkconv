#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

import os
import pathlib
import pandas as pd
import csv
use_mapping=False
if use_mapping:
    from label_dog import DogOpenImagesLabelNames as OpenImagesLabelNames
else:
    from label_default import OpenImagesLabelNames
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

_min_xmin = 9999999999
_min_ymin = 9999999999
_max_xmax = 0
_max_ymax = 0

def main():
    global _min_xmin, _min_ymin, _max_xmax, _max_ymax

    #####
    # input
    OpenImages_dir = '/data/huge/OpenImages'
    if use_mapping:
        OpenImagesLabelNames.init('/data/work/dog/00input-dog/dog.txt', '/data/huge/OpenImages/labels/class-descriptions-boxable.csv')
        assert 0 == OpenImagesLabelNames.label_index('/m/0bt9lr')
    else:
        OpenImagesLabelNames.init('/data/huge/OpenImages/labels/class-descriptions-boxable.csv')
        assert 0 <= OpenImagesLabelNames.label_index('/m/0bt9lr')
    splits = ( 'train', 'validation', 'test' )
    files = ( 'labels/{}-annotations-bbox.csv', 'labels/{}-annotations-human-imagelabels-boxable.csv' )

    #####
    # output
    if use_mapping:
        OUT_dir = '/data/work/dog/00input-dog'
    else:
        OUT_dir = '/data/huge/OpenImages/yolo'
    NAME='openimages'

    for d in ( 'images', 'labels', 'lists' ):
        if not os.path.exists(os.path.join(OUT_dir, d)):
            os.makedirs(os.path.join(OUT_dir, d), exist_ok=True)
    if not os.path.exists(os.path.join(OUT_dir, 'images', NAME)):
        os.symlink(os.path.join(OpenImages_dir, 'images'), os.path.join(OUT_dir, 'images', NAME))
    ignore_image_ids = {}
    for split in splits:
        os.makedirs(os.path.join(OUT_dir, 'labels', NAME, split), exist_ok=True)
        conf = pd.read_csv(os.path.join(OpenImages_dir, files[1].format(split)))
        print('confidence sources: {}'.format(conf['Source'].unique()))
        ids = conf.loc[(conf['Confidence']==0) & (conf['LabelName'].isin(OpenImagesLabelNames.label_names())), 'ImageID'].values.reshape(-1)
        ignore_image_ids[split] = ids
        print('Ignoring ids({}):{}'.format(split, ids))

    # create labels and lists
    prev_skip = ''
    with open(os.path.join(OUT_dir, 'lists', NAME + '.txt'), 'w') as flist:
        for split in splits:
            #ImageID,Source,LabelName,Confidence
            #000002b66c9c498e,verification,/m/014j1m,0
            #ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
            #000026e7ee790996,freeform,/m/07j7r,1,0.071905,0.145346,0.206591,0.391306,0,1,1,0,0
            bbox = os.path.join(OpenImages_dir, files[0].format(split))
            with open(bbox, 'r') as fbbox:
                lines = []
                lines_id = ''
                use_for_negative = False
                for row in csv.DictReader(fbbox):
                    if (use_for_negative or lines) and lines_id != row['ImageID']:
                        write_lines(flist, OUT_dir, NAME, split, lines_id, lines)
                        lines = []
                        lines_id = ''
                        use_for_negative = False
                    try:
                        label_index = OpenImagesLabelNames.label_index(row['LabelName'])
                        if 0 > label_index:
                            assert use_mapping
                            use_for_negative = True
                            lines_id = row['ImageID']
                            continue
                        if row['ImageID'] in ignore_image_ids[split]:
                            print('Skipping {}: Not verified...'.format(row['ImageID']))
                            continue
                    except:
                        assert use_mapping
                        if prev_skip != row['ImageID'][:2]:
                            prev_skip = row['ImageID'][:2]
                            print('Skipping {}:{}...'.format(split, prev_skip))
                        continue
                    xmin = float(row['XMin'])
                    ymin = float(row['YMin'])
                    xmax = float(row['XMax'])
                    ymax = float(row['YMax'])
                    if  _min_xmin > xmin:
                        _min_xmin = xmin
                    if  _min_ymin > ymin:
                        _min_ymin = ymin
                    if  _max_xmax < xmax:
                        _max_xmax = xmax
                    if  _max_ymax < ymax:
                        _max_ymax = ymax
                    assert 0 <= xmin < 1, bbox
                    assert 0 <= ymin < 1, bbox
                    assert 0 < xmax <= 1, bbox
                    assert 0 < ymax <= 1, bbox
                    line = '{} {:1.7f} {:1.7f} {:1.7f} {:1.7f}\n'.format(
                        label_index,
                        (xmin + xmax) / 2.0,
                        (ymin + ymax) / 2.0,
                        xmax - xmin,
                        ymax - ymin)
                    lines.append(line)
                    assert '' == lines_id or row['ImageID'] == lines_id, '{}, {}'.format(row['ImageID'], lines_id)
                    lines_id = row['ImageID']
                if use_for_negative or lines:
                    write_lines(flist, OUT_dir, NAME, split, lines_id, lines)

    print('min xmin: {:f}'.format(_min_xmin))
    print('min ymin: {:f}'.format(_min_ymin))
    print('max xmax: {:f}'.format(_max_xmax))
    print('max ymax: {:f}'.format(_max_ymax))

# =============================================================================
# =============================================================================

if __name__=='__main__':
    main()

# end of file
