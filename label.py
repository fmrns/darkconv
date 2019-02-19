#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

import threading
import csv

def _read_file_or_tuple(file_or_tuple, *, use_lower, expected_num, is_csv):
    if isinstance(file_or_tuple, (tuple, list)):
        label_names = file_or_tuple
    else:
        label_names = []
        with open(file_or_tuple, 'r', newline='') as f:
            if is_csv:
                for row in csv.reader(f):
                    if row:
                        label_names.append(row[0].lower() if use_lower else row[0])
            else:
                for line in f:
                    id_ = line.split()
                    if id_:
                        label_names.append(id_[0].lower() if use_lower else id_[0])
    if expected_num:
        assert expected_num == len(label_names), '{}, {}'.format(expected_num, len(label_names))
    return label_names

class LabelNames():
    _lock = threading.Lock()
    _label_names = None
    _use_lower = None

    def __new__(self):
        raise NotImplementedError('singleton')

    @classmethod
    def init(cls, file_or_tuple, *, expected_num=None, use_lower=False, is_csv=False):
        if cls._label_names is not None:
            assert False, 'init() called again.'
        super().__new__(cls)
        with cls._lock:
            if cls._label_names is not None:
                assert False, 'init() called again.'
            cls._label_names = _read_file_or_tuple(file_or_tuple, use_lower=use_lower, expected_num=expected_num, is_csv=is_csv)
            cls._use_lower = use_lower
            return cls

    @classmethod
    def save(cls, file):
        with open(file, 'w', newline='\n') as fo:
            for label in cls._label_names:
                fo.write('{}\n'.format(label))

    @classmethod
    def use_lower(cls):
        return cls._use_lower

    @classmethod
    def label_names(cls):
        return cls._label_names

    @classmethod
    def label_name(cls, i):
        return cls._label_names[i]

    @classmethod
    def label_index(cls, name):
        return cls._label_names.index(name.lower() if cls._use_lower else name)

class MappedLabelNames(LabelNames):
    _mapper = None
    _label_names_src = None
    _label_names_for_negative = None

    @classmethod
    def init(cls, label_dst, label_src, label_mapper=None, file_or_tuple_for_negative=None):
        if cls._label_names is not None:
            assert False, 'init() called again.'
        with cls._lock:
            if cls._label_names is not None:
                assert False, 'init() called again.'
            super().init(label_dst.label_names(), use_lower=label_dst.use_lower())
            cls._mapper = {} if label_mapper is None else label_mapper
            cls._label_names_src = label_src.label_names()
            cls._label_names_for_negative = set(_read_file_or_tuple(file_or_tuple_for_negative,
                    use_lower=label_dst.use_lower())) if file_or_tuple_for_negative else set()
            dup = set(cls._mapper.keys()) & set(cls._label_names_for_negative)
            assert 0 == len(dup), 'Both in mapper and negative: {}'.fomat(dup)
            return cls

    @classmethod
    def label_index(cls, name):
        nm = cls._label_names_src[name] if isinstance(name, int) else name
        nm = nm.lower() if cls._use_lower else nm
        nm = cls._mapper[nm] if nm in cls._mapper else nm
        return -1 if nm in cls._label_names_for_negative else cls._label_names.index(nm)

# end of file
