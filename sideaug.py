#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT
import os
import contextlib
import pathlib
import tempfile
import math
import random
import numpy as np
import cv2
from bbox import BBox
#random.seed(42)

class BBoxes():
    BACKGROUND_RANDOM = -1
    BACKGROUND_RANDOM_PLAIN = -2

    interps = ( cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, )

    @staticmethod
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

    def __init__(self, *, image=None, bboxes=None):
        self._image = cv2.imread(image)
        assert self._image is not None, image
        if image:
            _, self._image_ext = os.path.splitext(image)
        self._bboxes = bboxes
        for bbox in bboxes:
            bbox.set_size(hw=self._image.shape[:2])

    @property
    def image(self):
        return self._image

    @property
    def image_ext(self):
        return self._image_ext

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
        M, hw =  BBoxes.matrixRotate(self._image.shape[:2], rad, scale)
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

    def rotate_random(self, jitter_scale=.3):
        self.rotate(random.uniform(0, 2 * math.pi),
                    scale=1 + random.uniform(-jitter_scale, jitter_scale),
                    interpolation=random.choice(BBoxes.interps),
                    background=random.choice(( BBoxes.BACKGROUND_RANDOM, BBoxes.BACKGROUND_RANDOM_PLAIN )))

    def rotate_random90(self, right_angle, jitter_degree=5, jitter_scale=.3):
        jitter_rad = math.pi * jitter_degree / 180
        self.rotate(right_angle * math.pi / 2 + random.uniform(-jitter_rad, jitter_rad),
                    scale=1 + random.uniform(-jitter_scale, jitter_scale),
                    interpolation=random.choice(BBoxes.interps),
                    background=random.choice(( BBoxes.BACKGROUND_RANDOM, BBoxes.BACKGROUND_RANDOM_PLAIN )))

    def draw(self, bgr=(255, 255, 0), px=5):
        rc = self._image
        for bbox in self._bboxes:
            rc = bbox.draw(rc, bgr=bgr, px=px)
        return rc

def write_yolo(bboxes, fyolo):
    for bbox in bboxes.bboxes:
        yolo = bbox.get(BBox.YOLO)
        fyolo.write('{} {} {} {} {}\n'.format(bbox.label, *yolo))

def write_image_labels(bboxes, out_image_file, out_label_file):
    if os.path.exists(out_image_file):
        _, temp_image_file = tempfile.mkstemp(prefix='tmp.', suffix=bboxes.image_ext, dir=os.path.dirname(out_image_file))
        os.close(_)
        cv2.imwrite(temp_image_file, bboxes.image)

        with tempfile.NamedTemporaryFile(mode='w', newline='\n', delete=False,
                                         prefix='tmp.', suffix='.txt', dir=os.path.dirname(out_label_file)) as flabel:
            write_yolo(bboxes, flabel)
            temp_label_file = flabel.name

        os.replace(temp_image_file, out_image_file)
        os.replace(temp_label_file, out_label_file)
    else:
        cv2.imwrite(out_image_file, bboxes.image)
        with open(out_label_file, 'w') as flabel:
            write_yolo(bboxes, flabel)

def symlink_image_labels(orig_image_file, orig_label_file, out_image_file, out_label_file):
    if os.path.exists(out_image_file):
        _, temp_image_file = tempfile.mkstemp(prefix='tmp.', dir=os.path.dirname(out_image_file))
        os.close(_)
        os.remove(temp_image_file)
        os.symlink(os.path.relpath(orig_image_file, start=os.path.dirname(out_image_file)), temp_image_file)

        _, temp_label_file = tempfile.mkstemp(prefix='tmp.', dir=os.path.dirname(out_label_file))
        os.close(_)
        os.remove(temp_label_file)
        os.symlink(os.path.relpath(orig_label_file, start=os.path.dirname(out_label_file)), temp_label_file)

        os.replace(temp_image_file, out_image_file)
        os.replace(temp_label_file, out_label_file)
    else:
        os.symlink(os.path.relpath(orig_image_file, start=os.path.dirname(out_image_file)), out_image_file)
        os.symlink(os.path.relpath(orig_label_file, start=os.path.dirname(out_image_file)), out_label_file)

def aug_all(orig_image_dir, orig_label_dir, in_list_file, in_list_file_base, new_image_dir, new_label_dir, out_list_file,
            h_flip_prob, rotate90_prob):
    MODE_REPLACE_FILE = 0
    MODE_RENAME_NEW_DIR = 1
    MODE_RENAME_ORIG_RENAME_NEW_DIR = 2

    #orig, new   tmp file, replace file
    #orig        tmp dir, rename tmp->new
    #      new   tmp dir, rename new->orig, rename tmp->new

    if os.path.exists(orig_image_dir) and os.path.exists(new_image_dir):
        assert os.path.exists(new_label_dir), new_label_dir
        assert os.path.exists(orig_label_dir), orig_label_dir
        print('{} -> {}'.format(orig_image_dir, new_image_dir))
        mode = MODE_REPLACE_FILE
        in_image_dir  = orig_image_dir
        in_label_dir  = orig_label_dir
        out_image_dir = new_image_dir
        out_label_dir = new_label_dir
    else:
        if os.path.exists(orig_image_dir):
            assert os.path.exists(orig_label_dir), orig_label_dir
            print('{} -> {}(tmp)'.format(orig_image_dir, new_image_dir))
            mode = MODE_RENAME_NEW_DIR
            in_image_dir  = orig_image_dir
            in_label_dir  = orig_label_dir
        else:
            assert os.path.exists(new_image_dir), new_image_dir
            assert os.path.exists(new_label_dir), new_label_dir
            print('{}({})-> {}(tmp)'.format(new_image_dir, orig_image_dir, new_image_dir))
            mode = MODE_RENAME_ORIG_RENAME_NEW_DIR
            in_image_dir  = new_image_dir
            in_label_dir  = new_label_dir
        os.makedirs(os.path.dirname(new_image_dir), exist_ok=True)
        os.makedirs(os.path.dirname(new_label_dir), exist_ok=True)
        out_image_dir = tempfile.mkdtemp(prefix='tmp.', dir=os.path.dirname(new_image_dir))
        out_label_dir = tempfile.mkdtemp(prefix='tmp.', dir=os.path.dirname(new_label_dir))

    with open(in_list_file, 'r', newline='\n') as in_image_list, \
         open(out_list_file, 'w', newline='\n') if out_list_file else contextlib.nullcontext() as out_image_list:
        for image_file in in_image_list:
            if image_file.endswith('\n'):
                image_file = image_file[:-1]
            rel_image_file = os.path.relpath(image_file, start=in_list_file_base)
            assert '..' not in rel_image_file, '{},{},{}'.format(image_file, new_image_dir, rel_image_file)
            rel_label_file, image_ext = os.path.splitext(rel_image_file)
            rel_label_file += '.txt'
            in_image_file = os.path.join(in_image_dir, rel_image_file)
            in_label_file = os.path.join(in_label_dir, rel_label_file)
            out_image_file = os.path.join(out_image_dir, rel_image_file)
            out_label_file = os.path.join(out_label_dir, rel_label_file)
            print('Processing {}'.format(rel_image_file))
            if out_list_file:
                out_image_list.write('{}\n'.format(pathlib.PurePath(os.path.join(new_image_dir, rel_image_file)).as_posix()))
            os.makedirs(os.path.dirname(out_image_file), exist_ok=True)
            os.makedirs(os.path.dirname(out_label_file), exist_ok=True)
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
                do_h_flip = h_flip_prob   >= random.random()
                do_rotate = rotate90_prob >= random.random()
                if do_h_flip or do_rotate:
                    bboxes = BBoxes(image=in_image_file, bboxes=bboxes)
                    if do_h_flip: bboxes.h_flip()
                    if do_rotate:
                        if 1 > len(bboxes.bboxes):
                            bboxes.rotate_random()
                        else:
                            bboxes.rotate_random90(random.randrange(4))
                    write_image_labels(bboxes, out_image_file, out_label_file)
                else:
                    symlink_image_labels(os.path.join(orig_image_dir, rel_image_file),
                                         os.path.join(orig_label_dir, rel_label_file),
                                         out_image_file, out_label_file)

    if MODE_REPLACE_FILE == mode:
        pass
    else:
        if MODE_RENAME_ORIG_RENAME_NEW_DIR == mode:
            print('{} -> {}'.format(new_image_dir, orig_image_dir))
            os.rename(new_image_dir, orig_image_dir)
            print('{} -> {}'.format(new_label_dir, orig_label_dir))
            os.rename(new_label_dir, orig_label_dir)
        else:
            assert MODE_RENAME_NEW_DIR == mode

        print('{} -> {}'.format(out_image_dir, new_image_dir))
        os.rename(out_image_dir, new_image_dir)
        print('{} -> {}'.format(out_label_dir, new_label_dir))
        os.rename(out_label_dir, new_label_dir)

if __name__=='__main__':
    #os.chdir('m:/data/work/dog/00input-chihuahua')
    params = {
        'orig_image_dir': 'images.orig/G',
        'orig_label_dir': 'labels.orig/G',
          'in_list_file': 'lists/G.txt',
     'in_list_file_base': 'images/G',

         'new_image_dir': 'images/G.0',
         'new_label_dir': 'labels/G.0',
         'out_list_file': 'lists/G.0.txt',

         'h_flip_prob'  : 0.0,
         'rotate90_prob': 0.0,
    }
    #aug_all(**params)
    
    params['h_flip_prob']   = 0.1
    params['rotate90_prob'] = 1.0
    for i in range(1, 10):
        params['new_image_dir'] = 'images/G.{}'.format(i)
        params['new_label_dir'] = 'labels/G.{}'.format(i)
        params['out_list_file'] = 'lists/G.{}.txt'.format(i)
        aug_all(**params)

    for name in ( 'coco', 'ilsvrc', 'openimages', 'voc2012' ):
        params = {
            'orig_image_dir': 'images.orig/' + name,
            'orig_label_dir': 'labels.orig/' + name,
              'in_list_file': 'lists/' + name + '.txt',
         'in_list_file_base': 'images/' + name,
    
             'new_image_dir': 'images/' + name,
             'new_label_dir': 'labels/' + name,
             'out_list_file': None,
    
             'h_flip_prob'  : 0.1,
             'rotate90_prob': 1.0,
        }
        aug_all(**params)

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
