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
            return
        super().__new__(cls)
        with cls._lock:
            if cls._label_names is not None:
                assert False, 'init() called again.'
                return
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
    _label_src = None
    _label_dst = None
    _label_names_for_negative = None

    @classmethod
    def init(cls, label_dst, label_src, label_mapper=None, file_or_tuple_for_negative=None):
        super().init(label_src.label_names(), use_lower=label_src.use_lower())

        cls._mapper = {} if label_mapper is None else label_mapper
        cls._label_src = label_src
        cls._label_dst = label_dst
        cls._label_names_for_negative = set(_read_file_or_tuple(file_or_tuple_for_negative,
                use_lower=label_src.use_lower(), expected_num=None, is_csv=False)) if file_or_tuple_for_negative else set()

        assert set(cls._mapper.keys()) <= set(cls._label_names), \
            'Invalid mapper keys: {}'.fomat(set(cls._mapper.keys()) - set(cls._label_names))

        assert set(cls._label_names_for_negative) <= set(cls._label_names), \
            'Invalid negative keys: {}'.fomat(set(cls._label_names_for_negative) - set(cls._label_names))

        dup = set(cls._mapper.keys()) & set(cls._label_names_for_negative)
        assert 0 == len(dup), 'Both in mapper and negative: {}'.fomat(dup)

        mapper_values = []
        for val in cls._mapper.values():
            if isinstance(val, (tuple, list)):
                mapper_values.extend(val)
            elif val:
                mapper_values.append(val)
        assert set(label_dst.label_names()) >= set(mapper_values), \
            'Invalid mapped values: {}'.format(set(mapper_values) - set(label_dst.label_names()))
        for src_label in label_src.label_names():
            try:
                dst_index = cls.label_index(src_label)
            except:
                print('{}(skip),'.format(src_label), end='', flush=True)
                continue
            if isinstance(dst_index, (tuple, list)):
                print('{}->{},'.format(src_label, tuple(cls.label_name_dst(x) for x in dst_index)))
            elif 0 <= dst_index:
                print('{}->{},'.format(src_label, cls.label_name_dst(dst_index)))
            else:
                print('{}(neg),'.format(src_label))
        return cls

    @classmethod
    def label_names_for_negative(cls):
        return cls._label_names_for_negative

    @classmethod
    def label_names_dst(cls):
        return cls._label_dst.label_names()

    @classmethod
    def label_name_dst(cls, i):
        return cls._label_dst.label_name(i)

    @classmethod
    def label_index_src(cls, src_label):
        return cls._label_src.label_index(src_label)

    @classmethod
    def label_index_dst(cls, dst_label):
        return cls._label_dst.label_index(dst_label)

    # src label -> dst index
    @classmethod
    def label_index(cls, src_label):
        if isinstance(src_label, int):
            nm = cls.label_name(src_label)
        else:
            nm = src_label.lower() if cls._use_lower else src_label
        if nm in cls._label_names_for_negative:
            return -1
        if nm in cls._mapper:
            nm = cls._mapper[nm]
            if nm is None:
                # same exception which list.index() raises.
                raise ValueError('Discarding the label: {}'.format(src_label))
            if isinstance(nm, (tuple, list)):
                return tuple(cls.label_index_dst(n) for n in nm)
        return cls.label_index_dst(nm)

# end of file
