#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

from label import LabelNames

class OpenImagesLabelNames(LabelNames):
    @classmethod
    def init(cls, file=None, is_csv=True):
        super().init('/data/huge/OpenImages/labels/class-descriptions-boxable.csv' if file is None else file, expectes_num=601)
        assert '/m/0bt9lr' in cls.label_names()
        assert '/m/0306r'  in cls.label_names()
        return cls

class ILSVRCLabelNames(LabelNames):
    @classmethod
    def init(cls, file=None):
        #-----------------------------------
        # The mapping between the 1000 synset id and their descriptions.
        # For example, Line 1 says n01440764 tench, Tinca tinca means this is class 1,
        # has a synset id of n01440764, and it contains the fish tench.
        #-----------------------------------
        # -> need +1 for submission
        super().init('/data/huge/ILSVRC/LOC_synset_mapping.txt' if file is None else file, expected_num=1000)
        assert 'n02085620' in cls.label_names()
        assert 'n02085782' in cls.label_names()
        assert 'n02088364' in cls.label_names()
        return cls

class COCOLabelNames(LabelNames):
    @classmethod
    def init(cls, file=None):
        super().init('/opt/darknet/data/coco.names' if file is None else file, expected_num=80)
        assert 'dog' in cls.label_names()
        return cls

class VOCLabelNames(LabelNames):
    @classmethod
    def init(cls, file=None):
        super().init('/opt/darknet/data/voc.names' if file is None else file, expected_num=20, use_lower=True)
        assert 'dog' in cls.label_names()
        assert 'cow' in cls.label_names()
        return cls

# end of file
