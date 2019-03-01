#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT
import os
import tempfile
import math
import random
import numpy as np
import cv2
from bbox import BBox
random.seed(42)

def matrixRotate(hw, rad, scale=1):
    a = scale * math.cos(rad)
    b = scale * math.sin(rad)
    h, w = hw
    assert 1 <= w
    assert 1 <= h
    R = np.array(
         ( (  a, b, ),
           ( -b, a, ), ))
    # (w-1, 0), (0, h-1), (w-1, h-1)
    points = np.array( 
       ( ( w-1,   0 ),
         (   0, h-1 ),
         ( w-1, h-1 ), ))
    rp = R.dot(points.T)
    xmin = min(0, min(rp[0]))
    xmax = max(0, max(rp[0]))
    ymin = min(0, min(rp[1]))
    ymax = max(0, max(rp[1]))
    n_w_1 = xmax - xmin
    n_h_1 = ymax - ymin
    min_wh_1 = scale * (min(w, h) - 1)
    sqr_wh_1 = scale ** 2 * ((h-1) ** 2 + (w-1) ** 2)
    assert min_wh_1 <= n_w_1, '{}, {}'.format(min_wh_1, n_w_1)
    assert n_w_1 ** 2 <= sqr_wh_1
    assert min_wh_1 <= n_h_1, '{}, {}'.format(min_wh_1, n_h_1)
    assert n_h_1 ** 2 <= sqr_wh_1
    M = np.array(
        ( (  a, b, - xmin ),
          ( -b, a, - ymin ), ))
    return M, ( 1 + math.ceil(n_h_1), 1 + math.ceil(n_w_1) )

interps = ( cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, )
class BBoxes():
    BACKGROUND_RANDOM = -1
    BACKGROUND_RANDOM_PLAIN = -2
    
    def __init__(self, *, image=None, bboxes=None):
        self._image = image
        self._bboxes = bboxes
        for bbox in bboxes:
            bbox.set_size(hw=image.shape[:2])
    
    @property
    def image(self):
        return self._image
    
    @property
    def bboxes(self):
        return self._bboxes

    def h_flip(self):
        self._image = cv2.flip(self._image, 1)
        for bbox in self._bboxes:
            bbox.h_flip()

    def v_flip(self):
        self._image = cv2.flip(self._image, 0)
        for bbox in self._bboxes:
            bbox.v_flip()

    def rotate(self, rad, scale, interpolation, background):
        M, hw =  matrixRotate(self._image.shape[:2], rad, scale)
        c = self._image.shape[2] if len(self._image.shape) > 2 else 1
        if background == BBoxes.BACKGROUND_RANDOM:
            hwc_back = np.random.randint(0, 256, (hw[0], hw[1], c), np.uint8)
        else:
            hwc_back = np.zeros((hw[0], hw[1], 3), np.uint8)
            if background == BBoxes.BACKGROUND_RANDOM_PLAIN:
                v = []
                for _ in range(c):
                    v.append(random.randrange(0, 256))
                hwc_back[:] = v[0] if 1 == c else v
            elif background:
                hwc_back[:] = background
        self._image = cv2.warpAffine(self._image, M, hw[::-1], hwc_back,
                                     flags=interpolation, borderMode=cv2.BORDER_TRANSPARENT)
        for bbox in self._bboxes:
            bbox.warpAffine(M, hw[::-1])
        
    def rotate90(self, right_angle, jitter_degree=5, jitter_scale=.3):
        jitter_rad = math.pi * jitter_degree / 180
        self.rotate(right_angle * math.pi / 2 + random.uniform(-jitter_rad, jitter_rad),
                    scale=1 + random.uniform(-jitter_scale, jitter_scale),
                    interpolation=random.choice(interps),
                    background=random.choice(( BBoxes.BACKGROUND_RANDOM, BBoxes.BACKGROUND_RANDOM_PLAIN )))
    
    def draw(self, bgr=(255, 255, 0), px=5):
        rc = self._image
        for bbox in self._bboxes:
            rc = bbox.draw(rc, bgr=bgr, px=px)
        return rc

if __name__=='__main__':
    # read only files.
    #orig_image_dir = 'images.orig'
    #orig_label_dir = 'labels.orig'
    orig_image_dir = 'images.orig'
    orig_label_dir = 'labels.orig'
    
    # new_xxx_dir's are the directories, listed in the image list.
    new_image_dir  = 'images'
    new_label_dir  = 'labels'
    # image list
    train_txt = '/data/work/dog/00input-chihuahua/lists/.all.txt'

    if os.path.isdir(orig_image_dir):
        in_image_dir  = orig_image_dir
        out_image_dir = new_image_dir
    else:
        in_image_dir  = new_image_dir
        out_image_dir = tempfile.mkdtemp(prefix='tmp.', dir=os.path.dirname(new_image_dir))
    if os.path.isdir(orig_label_dir):
        in_label_dir  = orig_label_dir
        out_label_dir = new_label_dir
    else:
        in_label_dir  = new_label_dir
        out_label_dir = tempfile.mkdtemp(prefix='tmp.', dir=os.path.dirname(new_label_dir))
    assert not os.path.exists(orig_image_dir) or not os.path.samefile(out_image_dir, orig_image_dir), '{}, {}'.format(out_image_dir, orig_image_dir)
    assert not os.path.exists(orig_label_dir) or not os.path.samefile(out_label_dir, orig_label_dir), '{}, {}'.format(out_label_dir, orig_label_dir)
    
    with open(train_txt, 'r', newline='\n') as image_list:
        for image_file in image_list:
            if image_file.endswith('\n'):
                image_file = image_file[:-1]
            rel_image_file = os.path.relpath(image_file, start=new_image_dir)
            rel_label_file, image_ext = os.path.splitext(rel_image_file)
            rel_label_file += '.txt'
            in_image_file = os.path.join(in_image_dir, rel_image_file)
            in_label_file = os.path.join(in_label_dir, rel_label_file)
            out_image_file = os.path.join(out_image_dir, rel_image_file)
            out_label_file = os.path.join(out_label_dir, rel_label_file)
            
            print('Processing {}'.format(rel_image_file))
            os.makedirs(os.path.dirname(out_image_file), exist_ok=True)
            os.makedirs(os.path.dirname(out_label_file), exist_ok=True)
            assert not os.path.exists(out_image_file) or not os.path.samefile(in_image_file, out_image_file), '{}, {}'.format(in_image_file, out_image_file)
            assert not os.path.exists(out_label_file) or not os.path.samefile(in_label_file, out_label_file), '{}, {}'.format(in_label_file, out_label_file)
            with open(in_label_file, 'r') as flabel:
                bboxes = []
                for lline in flabel:
                    vals = lline.split()
                    if not vals:
                        continue
                    try:
                        label, *bbox = vals
                        bbox = tuple(map(float, bbox))
                        bboxes.append(BBox(type_=BBox.YOLO, bbox=bbox, label=label))
                    except:
                        print('Warning: invalid line: {}'.format(lline))
                        raise
                bboxes = BBoxes(image=cv2.imread(in_image_file), bboxes=bboxes)
                if 0.1 > random.random():
                    bboxes.h_flip()
                if 0.5 > random.random():
                    bboxes.rotate90(random.randrange(4))
                temp_image_file = None
                temp_label_file = None
                try:
                    _, temp_image_file = tempfile.mkstemp(prefix='tmp.', suffix=image_ext, dir=os.path.dirname(out_image_file))
                    os.close(_)
                    cv2.imwrite(temp_image_file, bboxes.image)
                    with tempfile.NamedTemporaryFile(mode='w+', newline='\n', delete=False,
                                                     prefix='tmp.', suffix='.txt', dir=os.path.dirname(out_label_file)) as temp_flabel:
                        temp_label_file = temp_flabel.name
                        for bbox in bboxes.bboxes:
                            yolo = bbox.get(BBox.YOLO)
                            temp_flabel.write('{} {} {} {} {}\n'.format(bbox.label, *yolo))
                    os.replace(temp_image_file, out_image_file)
                    temp_image_file = None
                    os.replace(temp_label_file, out_label_file)
                    temp_label_file = None
                except Exception as e:
                    print('{}: {}'.format(type(e).__name__, e.args))
                    if temp_image_file and os.path.exists(temp_image_file):
                        os.remove(temp_image_file)
                    if temp_label_file and os.path.exists(temp_label_file):
                        os.remove(temp_label_file)

    if out_image_dir != new_image_dir:
        print('{} -> {}'.format(new_image_dir, orig_image_dir))
        os.rename(new_image_dir, orig_image_dir)
        print('{} -> {}'.format(out_image_dir, new_image_dir))
        os.rename(out_image_dir, new_image_dir)
    if out_label_dir != new_label_dir:
        print('{} -> {}'.format(new_label_dir, orig_label_dir))
        os.rename(new_label_dir, orig_label_dir)
        print('{} -> {}'.format(out_label_dir, new_label_dir))
        os.rename(out_label_dir, new_label_dir)
            
# =============================================================================
#     import matplotlib.pyplot as plt
#     #bgr = cv2.imread('/data/huge/AI/G/images/DSC_0008.JPG', cv2.IMREAD_COLOR)
#     bgr = cv2.imread('M:/CloudStation/photo/AI/G/images/DSC_0008.JPG', cv2.IMREAD_COLOR)
#     vals = []
#     #with open('/data/huge/AI/G/labels/DSC_0008.txt') as fi:
#     with open('M:/CloudStation/photo/AI/G/labels/DSC_0008.txt') as fi:
#         for line in fi:
#             vals.append(BBox(type_=BBox.YOLO, bbox=map(float, line.split()[1:])))
#     bboxes = BBoxes(image=bgr, bboxes=vals)
#     plt.imshow(cv2.cvtColor(bboxes.draw(), cv2.COLOR_BGR2RGB))
#     plt.show()
# =============================================================================
#    for i in range(30):
#        n_bboxes = bboxes.rotate90(random.randrange(4))
#        plt.imshow(cv2.cvtColor(n_bboxes.draw(), cv2.COLOR_BGR2RGB))
#        plt.show()

# end of file
