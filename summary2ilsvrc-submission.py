#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

# convert summary to ILSVRC submission file.
# summary file: CSV: confidence, VOC style BBOX(float*4), label

import os
import math
import pathlib
import argparse
import pandas as pd
from bbox import BBox
#from label_chihuahua import ChihuahuaLabelNames as LabelNames
#predefined_labels = '/data/work/dog/00input-chihuahua/chihuahua.txt'
from label_default import ILSVRCLabelNames as LabelNames
predefined_labels = '/data/huge/ILSVRC/LOC_synset_mapping.txt'

def main(args):
    global predefined_labels
    global _min_x_w, _min_y_h, _max_x_w, _max_y_h
    global _max_w, _max_h, _max_w_h, _max_h_w

    LabelNames.init(predefined_labels)

    print('Reading: {}'.format(args.test_list))
    with open(args.test_list, 'r') as ftest, open(args.output_file, 'w') as fo:
        fo.write('ImageId,PredictionString\n')
        for line in ftest:
            id_ = pathlib.PurePath(line).stem
            pred = ''
            summary = os.path.join(args.summary_dir, id_ + '.csv')
            if not os.path.exists(summary):
                print('***** CAUTION *****: no summary for {}'.format(id_))
            else:
                df = pd.read_csv(os.path.join(args.summary_dir, id_ + '.csv'), header=None)
                df.columns = [ 'confidence', 'voc_xmin', 'voc_ymin', 'voc_xmax', 'voc_ymax', 'label' ]
                df.sort_values(by='confidence', ascending=False, inplace=True)
                df = df.head(5)
                bboxes = []
                for index, row in df.iterrows():
                    if 1 > len(bboxes) or float(row[0]) >= args.threshold:
                        bboxes.append(BBox(type_=BBox.VOC, bbox=(tuple(map(float, row[1:5]))), label=row[5]))
                for b in bboxes:
                    bbox = b.get(type_=BBox.ILSVRC)
                    #bbox = b.get(type_=BBox.VOC)
                    if pred:
                        pred += ' '
                    # OC_synset_mapping.txt: The mapping between the 1000 synset id and their descriptions.
                    # For example, Line 1 says n01440764 tench, Tinca tinca means this is class 1, has a synset id of n01440764,
                    # and it contains the fish tench. 
                    pred += '{} {} {} {} {}'.format(
                        1 + LabelNames.label_index(b.label),
                        #LabelNames.label_index(b.label),
                        math.floor(bbox[0]), math.floor(bbox[1]),
                        math.ceil(bbox[2]),  math.ceil(bbox[3]))
            fo.write('{},{}\n'.format(id_, pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert summary to ILSVRC submission.')
    parser.add_argument('--summary-dir', '-s', default='summary', help='dirctory that contains ID + .csv files.')
    parser.add_argument('--test-list',   '-t', default='/data/huge/ILSVRC/yolo/lists/test.txt')
    parser.add_argument('--output-file', '-o', default='summary/submission.csv')
    parser.add_argument('--threshold',   '-c', default='0.0', type=float, help='threshold of confidence.')
    
    args = parser.parse_args() 
    main(args)

# end of file
