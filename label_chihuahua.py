#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Abacus Technologies, Inc.
# Copyright (c) 2019 Fumiyuki Shimizu
# MIT License: https://opensource.org/licenses/MIT

from label import LabelNames, MappedLabelNames
from label_default import OpenImagesLabelNames, COCOLabelNames, ILSVRCLabelNames, VOCLabelNames

class ChihuahuaLabelNames(LabelNames):
    @classmethod
    def init(cls, file=None):
        super().init(file, expected_num=3)
        assert 'dog' in cls.label_names(), cls.label_names()
        assert 'chihuahua' in cls.label_names(), cls.label_names()
        assert 0 == cls.label_index('dog')
        assert 1 == cls.label_index('chihuahua')
        assert 2 == cls.label_index('G')
        return cls

class GLabelNames(LabelNames):
    @classmethod
    def init(cls, file=None):
        super().init(file, expected_num=1)
        return cls

class ChihuahuaGLabelNames(MappedLabelNames):
    mapper = { 'G': ( 'G', 'chihuahua', 'dog', ) }

    @classmethod
    def init(cls, fileChihuahua=None, fileG=None):
        super().init(ChihuahuaLabelNames.init(fileChihuahua), GLabelNames.init(fileG),
              label_mapper=cls.mapper)
        assert 'G' in cls.label_names()
        assert set(( 2, 1, 0 )) == set(cls.label_index('G'))
        return cls

class ChihuahuaVOCLabelNames(MappedLabelNames):
    mapper = { 'dog': None, } # chihuahua is not separated
    _neg = ( 'cat', 'cow', 'horse', 'sheep', )

    @classmethod
    def init(cls, fileChihuahua=None, fileVOC=None):
        super().init(ChihuahuaLabelNames.init(fileChihuahua), VOCLabelNames.init(fileVOC),
              label_mapper=cls.mapper, file_or_tuple_for_negative=cls._neg)
        assert 'dog' in cls.label_names()
        assert -1 == cls.label_index('cat')
        return cls

class ChihuahuaCOCOLabelNames(MappedLabelNames):
    mapper = { 'dog': None, } # chihuahua is not separated
    _neg = ( 'cat', 'horse', 'sheep', 'cow', 'elephant', 'zebra', 'giraffe', )

    @classmethod
    def init(cls, fileChihuahua=None, fileCOCO=None):
        super().init(ChihuahuaLabelNames.init(fileChihuahua), COCOLabelNames.init(fileCOCO),
              label_mapper=cls.mapper, file_or_tuple_for_negative=cls._neg)
        assert 'dog' in cls.label_names()
        assert -1 == cls.label_index('cat')
        return cls

class ChihuahuaOpenImagesLabelNames(MappedLabelNames):
    mapper = { '/m/0bt9lr': None, }	# Dog. chihuahua is not separated
    _neg = (
        '/m/01yrx',	# Cat
        '/m/0306r',	# Fox
        '/m/03k3r',	# Horse
    )

    @classmethod
    def init(cls, fileChihuahua=None, fileOpenImages=None):
        super().init(ChihuahuaLabelNames.init(fileChihuahua), OpenImagesLabelNames.init(fileOpenImages),
              label_mapper=cls.mapper, file_or_tuple_for_negative=cls._neg)
        assert '/m/0bt9lr' in cls.label_names()
        assert -1 == cls.label_index('/m/03k3r')
        return cls

class ChihuahuaILSVRCLabelNames(MappedLabelNames):
    mapper = { 'n02085620': ( 'chihuahua', 'dog', ) }
    _dog = (
        'n02085782',	# Japanese spaniel
        'n02085936',	# Maltese dog, Maltese terrier, Maltese
        'n02086079',	# Pekinese, Pekingese, Peke
        'n02086240',	# Shih-Tzu
        'n02086646',	# Blenheim spaniel
        'n02086910',	# papillon
        'n02087046',	# toy terrier
        'n02087394',	# Rhodesian ridgeback
        'n02088094',	# Afghan hound, Afghan
        'n02088238',	# basset, basset hound
        'n02088364',	# beagle
        'n02088466',	# bloodhound, sleuthhound
        'n02088632',	# bluetick
        'n02089078',	# black-and-tan coonhound
        'n02089867',	# Walker hound, Walker foxhound
        'n02089973',	# English foxhound
        'n02090379',	# redbone
        'n02090622',	# borzoi, Russian wolfhound
        'n02090721',	# Irish wolfhound
        'n02091032',	# Italian greyhound
        'n02091134',	# whippet
        'n02091244',	# Ibizan hound, Ibizan Podenco
        'n02091467',	# Norwegian elkhound, elkhound
        'n02091635',	# otterhound, otter hound
        'n02091831',	# Saluki, gazelle hound
        'n02092002',	# Scottish deerhound, deerhound
        'n02092339',	# Weimaraner
        'n02093256',	# Staffordshire bullterrier, Staffordshire bull terrier
        'n02093428',	# American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier
        'n02093647',	# Bedlington terrier
        'n02093754',	# Border terrier
        'n02093859',	# Kerry blue terrier
        'n02093991',	# Irish terrier
        'n02094114',	# Norfolk terrier
        'n02094258',	# Norwich terrier
        'n02094433',	# Yorkshire terrier
        'n02095314',	# wire-haired fox terrier
        'n02095570',	# Lakeland terrier
        'n02095889',	# Sealyham terrier, Sealyham
        'n02096051',	# Airedale, Airedale terrier
        'n02096177',	# cairn, cairn terrier
        'n02096294',	# Australian terrier
        'n02096437',	# Dandie Dinmont, Dandie Dinmont terrier
        'n02096585',	# Boston bull, Boston terrier
        'n02097047',	# miniature schnauzer
        'n02097130',	# giant schnauzer
        'n02097209',	# standard schnauzer
        'n02097298',	# Scotch terrier, Scottish terrier, Scottie
        'n02097474',	# Tibetan terrier, chrysanthemum dog
        'n02097658',	# silky terrier, Sydney silky
        'n02098105',	# soft-coated wheaten terrier
        'n02098286',	# West Highland white terrier
        'n02098413',	# Lhasa, Lhasa apso
        'n02099267',	# flat-coated retriever
        'n02099429',	# curly-coated retriever
        'n02099601',	# golden retriever
        'n02099712',	# Labrador retriever
        'n02099849',	# Chesapeake Bay retriever
        'n02100236',	# German short-haired pointer
        'n02100583',	# vizsla, Hungarian pointer
        'n02100735',	# English setter
        'n02100877',	# Irish setter, red setter
        'n02101006',	# Gordon setter
        'n02101388',	# Brittany spaniel
        'n02101556',	# clumber, clumber spaniel
        'n02102040',	# English springer, English springer spaniel
        'n02102177',	# Welsh springer spaniel
        'n02102318',	# cocker spaniel, English cocker spaniel, cocker
        'n02102480',	# Sussex spaniel
        'n02102973',	# Irish water spaniel
        'n02104029',	# kuvasz
        'n02104365',	# schipperke
        'n02105056',	# groenendael
        'n02105162',	# malinois
        'n02105251',	# briard
        'n02105412',	# kelpie
        'n02105505',	# komondor
        'n02105641',	# Old English sheepdog, bobtail
        'n02105855',	# Shetland sheepdog, Shetland sheep dog, Shetland
        'n02106030',	# collie
        'n02106166',	# Border collie
        'n02106382',	# Bouvier des Flandres, Bouviers des Flandres
        'n02106550',	# Rottweiler
        'n02106662',	# German shepherd, German shepherd dog, German police dog, alsatian
        'n02107142',	# Doberman, Doberman pinscher
        'n02107312',	# miniature pinscher
        'n02107574',	# Greater Swiss Mountain dog
        'n02107683',	# Bernese mountain dog
        'n02107908',	# Appenzeller
        'n02108000',	# EntleBucher
        'n02108089',	# boxer
        'n02108422',	# bull mastiff
        'n02108551',	# Tibetan mastiff
        'n02108915',	# French bulldog
        'n02109047',	# Great Dane
        'n02109525',	# Saint Bernard, St Bernard
        'n02109961',	# Eskimo dog, husky
        'n02110063',	# malamute, malemute, Alaskan malamute
        'n02110185',	# Siberian husky
        'n02110341',	# dalmatian, coach dog, carriage dog
        'n02110627',	# affenpinscher, monkey pinscher, monkey dog
        'n02110806',	# basenji
        'n02110958',	# pug, pug-dog
        'n02111129',	# Leonberg
        'n02111277',	# Newfoundland, Newfoundland dog
        'n02111500',	# Great Pyrenees
        'n02111889',	# Samoyed, Samoyede
        'n02112018',	# Pomeranian
        'n02112137',	# chow, chow chow
        'n02112350',	# keeshond
        'n02112706',	# Brabancon griffon
        'n02113023',	# Pembroke, Pembroke Welsh corgi
        'n02113186',	# Cardigan, Cardigan Welsh corgi
        'n02113624',	# toy poodle
        'n02113712',	# miniature poodle
        'n02113799',	# standard poodle
        'n02113978',	# Mexican hairless
    )

    @classmethod
    def init(cls, fileChihuahua=None, fileILSVRC=None):
        for key in cls._dog:
            cls.mapper[key] = 'dog'
        super().init(ChihuahuaLabelNames.init(fileChihuahua), ILSVRCLabelNames.init(fileILSVRC), label_mapper=cls.mapper)
        assert 'dog' in cls.label_names_dst()
        assert set(( 1, 0 )) == set(cls.label_index('n02085620'))
        assert 0 == cls.label_index('n02085782')
        assert 0 == cls.label_index('n02113978')
        return cls

# end of file
