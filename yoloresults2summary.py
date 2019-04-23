#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

# summarize validation results of yolo

import os
import glob
import pathlib
import math
import argparse
from bbox import BBox
if True:
    from label_default import ILSVRCLabelNames as LabelNames
    predefined_labels = '/data/huge/ILSVRC/LOC_synset_mapping.txt'
else:
    from label_chihuahua import ChihuahuaLabelNames as LabelNames
    predefined_labels = '/data/work/dog/00input-chihuahua/chihuahua.txt'

def main(args):
    global predefined_labels
    global _min_x_w, _min_y_h, _max_x_w, _max_y_h
    global _max_w, _max_h, _max_w_h, _max_h_w

    LabelNames.init(predefined_labels)

    args.results_dir
    args.summary_dir

    os.makedirs(args.summary_dir, exist_ok=True)
    for f in glob.glob(pathlib.PurePath(args.summary_dir).as_posix() + '/*.csv'):
        os.remove(f)
    for label in LabelNames.label_names():
        print('Label: {}'.format(label))
        pred = None
        is_first = True
        for pred in glob.glob(pathlib.PurePath(args.results_dir).as_posix() + '/comp4_det_test*_' + label + '.txt'):
            if not is_first:
                raise ValueError('Multiple files exist for {}: {}'.format(label, pred))
            is_first = False
        if pred is None:                                    
            print('Skipping label: {}'.format(label))
            continue
        print('Reading: {}'.format(pred))
        with open(pred, 'r') as fpred:
            for line in fpred:
                id_, *flts = line.split()
                # confidence, VOC style values: 1<->width, 1<->height, 1<->width, 1<->height
                flts = tuple(map(float, flts))
                with open(os.path.join(args.summary_dir, id_ + '.csv'), 'a') as fsummary:
                    fsummary.write('{:1.15f},{:1.15f},{:1.15f},{:1.15f},{:1.15f},{}\n'.format(*flts,label))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='summarize validation results produced by darknet.')
    parser.add_argument('--results-dir', '-r', default='results')
    parser.add_argument('--summary-dir', '-s', required=True, help='output directory. *** existing files will be deleted.')
    args = parser.parse_args() 
    main(args)

# end of file
