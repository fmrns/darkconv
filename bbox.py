#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

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
    
    def __init__(self, *, hw=None, type_, bbox, eps=10e-2):
        if hw:
            self.h_1 = hw[0] - 1
            self.w_1 = hw[1] - 1
        if type_ == BBox.VOC or type_ == BBox.ILSVRC:
            self.xmin, self.ymin, self.xmax, self.ymax = (bbox[0]-1,bbox[1]-1,bbox[2]-1,bbox[3]-1) if type_ == BBox.VOC else bbox
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
        elif type_ == BBox.OPEN_IMAGES:
            self._xmin, self._ymin, self._xmax, self._ymax = bbox
            self.cx = (self._xmin + self._xmax) / 2.0
            self.cy = (self._ymin + self._ymax) / 2.0
            self.rw = self._xmax - self._xmin
            self.rh = self._ymax - self._ymin
            if hw:
                self.xmin = self._xmin * self.w_1
                self.xmax = self._xmax * self.w_1
                self.ymin = self._ymin * self.h_1
                self.ymax = self._ymax * self.h_1
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
            if hw:
                self.xmin = self.w_1 * self._xmin
                self.xmax = self.w_1 * self._xmax
                self.ymin = self.h_1 * self._ymin
                self.ymax = self.h_1 * self._ymax

        assert 0 < self.cx < 1, '{}'.format(self.cx)
        assert 0 < self.cy < 1, '{}'.format(self.cy)
        assert 0 < self.rw <= 1, '{}'.format(self.rw)
        assert 0 < self.rh <= 1, '{}'.format(self.rh)
        if hw:
            ixmin, iymin, ixmax, iymax = (round(self.xmin), round(self.ymin), round(self.xmax), round(self.ymax))
            self.xmin = self.prefer_small(self.xmin, ixmin, eps)
            self.ymin = self.prefer_small(self.ymin, iymin, eps)
            self.xmax = self.prefer_large(self.xmax, ixmax, eps, self.w_1)
            self.ymax = self.prefer_large(self.ymax, iymax, eps, self.h_1)

            assert 0 <= self.xmin < self.w_1, '{}, {}'.format(self.xmin, self.w_1)
            assert 0 <= self.ymin < self.h_1, '{}, {}'.format(self.ymin, self.h_1)
            assert 0 < self.xmax <= self.w_1, '{}, {}'.format(self.xmax, self.w_1)
            assert 0 < self.ymax <= self.h_1, '{}, {}'.format(self.ymax, self.h_1)
            assert self.xmin < self.xmax, '{}, {}'.format(self.xmin, self.xmax)
            assert self.ymin < self.ymax, '{}, {}'.format(self.ymin, self.ymax)

    def get(self, type_):
        if type_ == BBox.VOC:
            return (1+self.xmin, 1+self.ymin, 1+self.xmax, 1+self.ymax)
        if type_ == BBox.ILSVRC:
            return (self.xmin, self.ymin, self.xmax, self.ymax)
        if type_ == BBox.OPEN_IMAGES:
            return (self._xmin, self._ymin, self._xmax, self._ymax)
        assert type_ == BBox.YOLO, type_
        return (self.cx, self.cy, self.rw, self.rh)

    def h_flip(self):
        self.cx = 1 - self.cx
        self._xmin, self._xmax = ( 1 - self._xmax, 1 - self._xmin )
        if self.w_1:
            self.xmin, self.xmax = ( self.w_1 - self._xmax, self.w_1 - self._xmin )
        else:
            assert not hasattr(self, 'xmin')
            assert not hasattr(self, 'xmax')
        
    def v_flip(self):
        self.cy = 1 - self.cy
        self._ymin, self._ymax = ( 1 - self._ymax, 1 - self._ymin )
        if self.h_1:
            self.ymin, self.ymax = ( self.h_1 - self._ymax, self.h_1 - self._ymin )
        else:
            assert not hasattr(self, 'ymin')
            assert not hasattr(self, 'ymax')
    
# end of file
