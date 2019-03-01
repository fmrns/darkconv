#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT
import numpy as np
import cv2

class BBox():
    YOLO=0
    VOC=1
    ILSVRC=2
    OPEN_IMAGES=3
    
    h_1 = None 
    w_1 = None
    
    @staticmethod
    def prefer_small(p, newp, eps):
        dlt = newp - p
        return max(0, newp if dlt <= 0 or eps > dlt else newp - 1)

    @staticmethod
    def prefer_large(p, newp, eps, maxp):
        dlt = p - newp
        return min(maxp, newp if dlt <= 0 or eps > dlt else newp + 1)

    def is_absolute(self):
        return self._type in ( BBox.VOC, BBox.ILSVRC )
    
    def calc_absolute(self):
        if not self.is_absolute():
            self.xmin = self._xmin * self.w_1
            self.xmax = self._xmax * self.w_1
            self.ymin = self._ymin * self.h_1
            self.ymax = self._ymax * self.h_1

        ixmin, iymin, ixmax, iymax = ( round(self.xmin), round(self.ymin), round(self.xmax), round(self.ymax) )
        self.xmin = self.prefer_small(self.xmin, ixmin, self.eps)
        self.ymin = self.prefer_small(self.ymin, iymin, self.eps)
        self.xmax = self.prefer_large(self.xmax, ixmax, self.eps, self.w_1)
        self.ymax = self.prefer_large(self.ymax, iymax, self.eps, self.h_1)

        assert 0 <= self.xmin < self.w_1, '{}, {}'.format(self.xmin, self.w_1)
        assert 0 <= self.ymin < self.h_1, '{}, {}'.format(self.ymin, self.h_1)
        assert 0 < self.xmax <= self.w_1, '{}, {}'.format(self.xmax, self.w_1)
        assert 0 < self.ymax <= self.h_1, '{}, {}'.format(self.ymax, self.h_1)
        assert 0 <= self.xmin < self.xmax, '{}, {}'.format(self.xmin, self.xmax)
        assert 0 <= self.ymin < self.ymax, '{}, {}'.format(self.ymin, self.ymax)

    def calc_relative(self):
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
        assert 0 < self.cx < 1, '{}'.format(self.cx)
        assert 0 < self.cy < 1, '{}'.format(self.cy)
        assert 0 < self.rw <= 1, '{}'.format(self.rw)
        assert 0 < self.rh <= 1, '{}'.format(self.rh)
        assert 0 <= self._xmin < 1, '{}'.format(self._xmin)
        assert 0 <= self._ymin < 1, '{}'.format(self._ymin)
        assert 0 < self._xmax <= 1, '{},{},{}'.format(self._xmax, self.xmax, self.w_1)
        assert 0 < self._ymax <= 1, '{},{},{}'.format(self._ymax, self.ymax, self.h_1)
        
    def __init__(self, *, hw=None, type_, bbox, label=None, eps=10e-2):
        self._type = type_
        self._label = label
        self.eps = eps
        if self.is_absolute():
            if type_ == BBox.VOC:
                self.xmin, self.ymin, self.xmax, self.ymax = (bbox[0]-1,bbox[1]-1,bbox[2]-1,bbox[3]-1)
            else:
                assert type_ == BBox.ILSVRC
                self.xmin, self.ymin, self.xmax, self.ymax = bbox
            assert 0 <= self.xmin < self.xmax, '{}, {}'.format(self.xmin, self.xmax)
            assert 0 <= self.ymin < self.ymax, '{}, {}'.format(self.ymin, self.ymax)
        else:
            if type_ == BBox.OPEN_IMAGES:
                self._xmin, self._ymin, self._xmax, self._ymax = bbox
                self.cx = (self._xmin + self._xmax) / 2.0
                self.cy = (self._ymin + self._ymax) / 2.0
                self.rw = self._xmax - self._xmin
                self.rh = self._ymax - self._ymin
            else:
                assert type_ == BBox.YOLO, type_
                self.cx, self.cy, self.rw, self.rh = bbox
                # self.cx * 2 # xmax + xmin
                # self.rw     # xmax - xmin
                r2 = self.rw / 2.0
                self._xmin = self.cx - r2
                self._xmax = self.cx + r2
                r2 = self.rh / 2.0
                self._ymin = self.cy - r2
                self._ymax = self.cy + r2
            assert 0 < self.cx < 1, '{}'.format(self.cx)
            assert 0 < self.cy < 1, '{}'.format(self.cy)
            assert 0 < self.rw <= 1, '{}'.format(self.rw)
            assert 0 < self.rh <= 1, '{}'.format(self.rh)
        if hw:
            self.set_size(hw)

    def set_size(self, hw):
        assert self.h_1 is None
        assert self.w_1 is None
        self.h_1 = hw[0] - 1
        self.w_1 = hw[1] - 1
        # VOC(absolute): float -> int
        self.calc_absolute()
        if self.is_absolute():
            self.calc_relative()

    def get(self, type_):
        if type_ == BBox.VOC:
            return (1+self.xmin, 1+self.ymin, 1+self.xmax, 1+self.ymax)
        if type_ == BBox.ILSVRC:
            return (self.xmin, self.ymin, self.xmax, self.ymax)
        if type_ == BBox.OPEN_IMAGES:
            return (self._xmin, self._ymin, self._xmax, self._ymax)
        assert type_ == BBox.YOLO, type_
        return (self.cx, self.cy, self.rw, self.rh)

    @property
    def label(self):
        return self._label

    def h_flip(self):
        self.cx = 1 - self.cx
        self._xmin, self._xmax = ( 1 - self._xmax, 1 - self._xmin )
        if self.w_1:
            self.xmin, self.xmax = ( self.w_1 - self.xmax, self.w_1 - self.xmin )
        else:
            assert not hasattr(self, 'xmin')
            assert not hasattr(self, 'xmax')
        
    def v_flip(self):
        self.cy = 1 - self.cy
        self._ymin, self._ymax = ( 1 - self._ymax, 1 - self._ymin )
        if self.h_1:
            self.ymin, self.ymax = ( self.h_1 - self.ymax, self.h_1 - self.ymin )
        else:
            assert not hasattr(self, 'ymin')
            assert not hasattr(self, 'ymax')

    def warpAffine(self, M, wh):
        assert hasattr(self, 'xmin')
        assert hasattr(self, 'xmax')
        assert hasattr(self, 'ymin')
        assert hasattr(self, 'ymax')
        corners = np.array(
            ( ( self.xmin, self.ymin, 1 ),
              ( self.xmax, self.ymin, 1 ),
              ( self.xmin, self.ymax, 1 ),
              ( self.xmax, self.ymax, 1 ), ))
        n_corners = M.dot(corners.T)
        self.xmin = min(n_corners[0]).item()
        self.xmax = max(n_corners[0]).item()
        self.ymin = min(n_corners[1]).item()
        self.ymax = max(n_corners[1]).item()
        self.w_1 = wh[0] - 1
        self.h_1 = wh[1] - 1
        self.calc_relative()
        self.calc_absolute()
        
    def draw(self, cv2img, bgr=(255, 255, 0), px=5):
        h, w = cv2img.shape[:2]
        if self.h_1 or self.w_1:
            assert self.h_1 == h - 1, '{}, {}'.format(self.h_1, h - 1)
            assert self.w_1 == w - 1, '{}, {}'.format(self.w_1, w - 1)
        else:
            self.h_1 = h - 1
            self.w_1 = w - 1
            self.calc_int()
        return cv2.rectangle(np.copy(cv2img), (self.xmin, self.ymin) , (self.xmax, self.ymax), bgr, px)
    
# end of file
