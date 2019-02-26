#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT
import os
import cv2

class bbox():
    @staticmethod
    def prefer_small(p, newp, eps):
        dlt = newp - p
        if dlt < 0:
            return max(0, newp)
        if dlt == 0 or eps > dlt:
            return newp
        return max(0, newp - 1)

    @staticmethod
    def prefer_large(p, newp, eps, maxp):
        dlt = p - newp
        if dlt < 0:
            return min(maxp, newp)
        if dlt == 0 or eps > dlt:
            return newp
        return min(0, newp + 1)
    
    def __init__(self, *, hw, type_, bbox, eps=10e-2):
        self.h_1 = hw[0] - 1
        self.w_1 = hw[1] - 1
        if type_ == bbox.VOC or type_ == bbox.ILSVRC:
            self.xmin, self.ymin, self.xmax, self.ymax = (bbox[0]-1,bbox[1]-1,bbox[2]-1,bbox[3]-1) if type_ == bbox.VOC else bbox
            # 0, 1:
            #  width=2
            #  center = (0 + 1) / 2 / (2-1) = 0.5
            #  w = (1 - 0) / (2-1) = 1
            # 0, 2:
            #  width=3
            #  center = (0 + 2) / 2 / (3-1) = 0.5
            #  w = (2 - 0) / (3-1) = 1
            # 0, 3:
            #  width = 4
            #  center = (0 + 3) / 2 / (4-1) = 0.5
            #  w = (3 - 0) / (4-1) = 1
            self.cx = (self.xmin + self.xmax) / 2.0 / self.w_1
            self.cy = (self.ymin + self.ymax) / 2.0 / self.h_1
            self.rw = (self.xmax - self.xmin) / self.w_1
            self.rh = (self.ymax - self.ymin) / self.h_1
            self._xmin = self.xmin / self.w_1
            self._xmax = self.xmax / self.w_1
            self._ymin = self.ymin / self.h_1
            self._ymax = self.ymax / self.h_1
        elif type_ == bbox.OPEN_IMAGES:
            self._xmin, self._ymin, self._xmax, self._ymax = bbox
            self.cx = (self._xmin + self._xmax) / 2.0
            self.cy = (self._ymin + self._ymax) / 2.0
            self.rw = self._xmax - self._xmin
            self.rh = self._ymax - self._ymin
            self.xmin = self._xmin * self.w_1
            self.xmax = self._xmax * self.w_1
            self.ymin = self._ymin * self.h_1
            self.ymax = self._ymax * self.h_1
        else:
            assert type_ == bbox.YOLO, type_
            self.cx, self.cy, self.rw, self.rh = yolo
            # self.cx * 2 # xmax + xmin
            # self.rw     # xmax - xmin
            r2 = self.rw / 2.0
            self._xmin = self.cx - r2
            self._xmax = self.cx + r2
            r2 = self.rh / 2.0
            self._ymin = self.cy - r2
            self._ymax = self.cy + r2
            self.xmin = self.w_1 * self._xmin
            self.xmax = self.w_1 * self._xmax
            self.ymin = self.h_1 * self._ymin
            self.ymax = self.h_1 * self._ymax
        _xmin, _ymin, _xmax, _ymax = tuple(map(round, (self.xmin, self.ymin, self.xmax, self.ymax)))
        self.xmin = self.prefer_small(self.xmin, _xmin, eps)
        self.ymin = self.prefer_small(self.ymin, _ymin, eps)
        self.xmax = self.prefer_large(self.xmax, _xmax, eps, self.w_1)
        self.ymax = self.prefer_large(self.ymax, _ymax, eps, self.y_1)

        assert 0 <= self.xmin < self.w_1, '{}, {}'.format(self.xmin, self.w_1)
        assert 0 <= self.ymin < self.h_1, '{}, {}'.format(self.ymin, self.h_1)
        assert 0 < self.xmax <= self.w_1, '{}, {}'.format(self.xmax, self.w_1)
        assert 0 < self.ymax <= self.h_1, '{}, {}'.format(self.ymax, self.h_1)
        assert self.xmin < self.xmax, '{}, {}'.format(self.xmin, self.xmax)
        assert self.ymin < self.ymax, '{}, {}'.format(self.ymin, self.ymax)
        assert 0 < self.cx < 1, '{}'.format(self.cx)
        assert 0 < self.cy < 1, '{}'.format(self.cy)
        assert 0 < self.rw <= 1, '{}'.format(self.rw)
        assert 0 < self.rh <= 1, '{}'.format(self.rh)

    def get(self, type_):
        if type_ == bbox.VOC:
            return (1+self.xmin, 1+self.ymin, 1+self.xmax, 1+self.ymax)
        if type_ == bbox.ILSVRC:
            return (self.xmin, self.ymin, self.xmax, self.ymax)
        if type_ == bbox.OPEN_IMAGES:
            return (self._xmin, self._ymin, self._xmax, self._ymax)
        assert type_ == bbox.YOLO, type_
        return (self.cx, self.cy, self.rw, self.rh)

class bboxes():
    YOLO = 0
    VOC = 1
    ILSVRC = 2
    OPEN_IMAGES = 3
    
    def __init__(self, *, type_=bbox.VOC, img=None, bboxes=None):
        self.img = img
        img.shape[:2]
        
        def __init__(self, *, hw, type_, bbox, eps=10e-2):
        if type_ == bbox.VOC or type_ == bbox.ILSVRC:
            self.xmin, self.ymin, self.xmax, self.ymax = map(lambda n: n - 1, bbox) if type_ == bbox.VOC else bbox
            # 0, 1:
            #  width=2
            #  center = (0 + 1) / 2 / (2-1) = 0.5
            #  w = (1 - 0 + 1) / 2 = 1
            # 0, 2:
            #  width=3
            #  center = (0 + 2) / 2 / (3-1) = 0.5
            #  w = (2 - 0 + 1) / 3 = 1
            # 0, 3:
            #  width = 4
            #  center = (0 + 3) / 2 / (4-1) = 0.5
            #  w = (3 - 0 + 1) / 4 = 1
            self.cx = (self.xmin + self.xmax) / 2.0 / (self.w - 1)
            self.cy = (self.ymin + self.ymax) / 2.0 / (self.h - 1)
            self.rw = (self.xmax - self.xmin + 1) / self.w
            self.rh = (self.ymax - self.ymin + 1) / self.h
        elif type_ == bbox.OPEN_IMAGES:
            _xmin, _ymin, _xmax, _ymax = bbox
            self.cx = (_xmin + _xmax) / 2.0
            self.cy = (_ymin + _ymax) / 2.0
            self.rw = _xmax - _xmin
            self.rh = _ymax - _ymin
            self.xmin = _xmin * (w - 1)
            self.xmax = _xmax * (w - 1)
            self.ymin = _ymin * (h - 1)
            self.ymax = _ymax * (h - 1)
        else:
            assert type_ == bbox.YOLO, type_
            self.cx, self.cy, self.rw, self.rh = yolo
            _xx1 = self.cx * 2 * (self.w - 1) # xmax + xmin
            _xx2 = self.rw * self.w - 1       # xmax - xmin
            self.xmin = (_xx1 - _xx2) / 2.0
            self.xmax = (_xx1 + _xx2) / 2.0
            _yy1 = self.cy * 2 * (self.h - 1) # ymax + ymin
            _yy2 = self.rh * self.h - 1       # ymax - ymin
            self.ymin = (_yy1 - _yy2) / 2.0
            self.ymax = (_yy1 + _yy2) / 2.0
        assert 0 <= self.xmin < self.w, '{}, {}'.format(self.xmin, self.w)
        assert 0 <= self.ymin < self.h, '{}, {}'.format(self.ymin, self.h)
        assert 0 < self.xmax <= self.w, '{}, {}'.format(self.xmax, self.w)
        assert 0 < self.ymax <= self.h, '{}, {}'.format(self.ymax, self.h)
        assert 0 < self.cx < 1, '{}'.format(self.cx)
        assert 0 < self.cy < 1, '{}'.format(self.cy)
        assert 0 < self.rw <= 1, '{}'.format(self.rw)
        assert 0 < self.rh <= 1, '{}'.format(self.rh)
            
    def h_flip():
        img = cv2.flip(img, 1)
        self.cx = 1 - self.cx

    def v_flip():
        img = cv2.flip(img, 1)
        self.cx = 1 - self.cx
        
    def rotate(degree):
        img = cv2.flip(img, 1)
        self.cx = 1 - self.cx
        
        

interps = {
      'linear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
     'nearest': cv2.INTER_NEAREST,
       'cubic': cv2.INTER_CUBIC,
    'lanczos4': cv2.INTER_LANCZOS4,
}
def resize(img, width, height, interp):
    return cv2.resize(img, (width, height), interpolation=interp)

def h_flip(img, bbox):
    return cv2.resize(img, (width, height), interpolation=interp)


                            img = cv2.imread(img_file)
                            if img is None:
                                raise FileNotFoundError('[{}]'.format(img_file))
                            h, w = img.shape[:2]
                if not os.path.isfile(img_file):
                    raise FileNotFoundError('[{}]'.format(img_file))
                img = None
                for size in sizes:
                    ss = size_str(size['width'], size['height'])
                    for interp, v in interps.items():
                        img_file_out = os.path.join(out_base[ss][interp], img_file_rel)
                        parent_dir = os.path.dirname(img_file_out)
                        if parent_dir not in dirs:
                            print('{}'.format(parent_dir), flush=True)
                            os.makedirs(parent_dir, exist_ok=True)
                            dirs.append(parent_dir)
                        if os.path.isfile(img_file_out):
                            if 0 < os.path.getsize(img_file_out):
                                continue
                            os.remove(img_file_out)
                        if img is None:
                            if keep_aspect_ratio:
                                wr = size['width']  / w
                                hr = size['height'] / h
                                if hr >= wr:
                                    h = min(size['height'], round(h * size['width'] / w))
                                    w = size['width']
                                else:
                                    w = min(size['width'], round(w * size['height'] / h))
                                    h = size['height']
                            else:
                                w = size['width']
                                h = size['height']
                        cv2.imwrite(img_file_out, resize(img, v, w, h))
                        if not os.path.isfile(img_file_out):
                            raise FileNotFoundError('[{}]'.format(img_file_out))
                        #file_list_out = os.path.join(out_base[ss][interp], 'lists', '{}.txt'.format(file_list_in))
                        #trvl_list_out = os.path.join(out_base[ss][interp], 'lists', 'trainval.txt')
                        #os.makedirs(os.path.dirname(file_list_out), exist_ok=True)
                        #os.makedirs(os.path.dirname(img_file_out),  exist_ok=True)
                        #with open(file_list_out, 'a', newline='\n') as fout, \
                        #     open(trvl_list_out, 'a', newline='\n') if is_file_list_trainval else contextlib.nullcontext() as fout2:
                        #    fout.write('{}\n'.format(img_file_out))
                        #    if is_file_list_trainval:
                        #        fout2.write('{}\n'.format(img_file_out))

if __name__=='__main__':
    files = {
        'in_base': '/data/huge/ILSVRC/yolo',
        # in_base + '/lists'
        'list_in': ( 'train', 'val', 'test', ),
        # size, iterpolation
        'out_base': '/data/huge/ILSVRC-processed/ILSVRC.keep-ar.{}-{}',
    }
    resize_all(files,
               ({ 'width': 416, 'height': 416 },),
               keep_aspect_ratio=True)
    resize_all(files,
               ({ 'width': 608, 'height': 608 }, { 'width': 512, 'height': 512 },
                { 'width': 416, 'height': 416 }, { 'width': 300, 'height': 300 },),
               keep_aspect_ratio=True)

    files = {
        'in_base': '/data/huge/ILSVRC/yolo',
        # in_base + '/lists'
        'list_in': ( 'train', 'val', 'test', ),
        # size, iterpolation
        'out_base': '/data/huge/ILSVRC-processed/ILSVRC.{}-{}',
    }
    resize_all(files,
               ({ 'width': 416, 'height': 416 },))
    resize_all(files,
               ({ 'width': 608, 'height': 608 }, { 'width': 512, 'height': 512 },
                { 'width': 416, 'height': 416 }, { 'width': 300, 'height': 300 },))

# end of file
