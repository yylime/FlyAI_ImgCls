#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/8 15:40
# @Author  : yulin
# @E-mail  : 844202100@qq.com 
# @School  : bupt
# @File    : config.py
from flyai_sdk import MODEL_PATH

all_label_list = ['beach', 'circularfarmland', 'cloud', 'desert', 'forest', 'mountain',
                  'rectangularfarmland', 'residential', 'river', 'snowberg']

class Config:
    def __init__(self):
        self.lr = 0.0001
        self.img_size = 224
        self.bs = 32
        self.epochs = 25
        self.num_class = len(all_label_list)
        self.split = 0.02
        # self.model_names = ['convnext_tiny']
        self.model_names = ['efficientnet-b4', 'resnet50_cbam', 'resnext50_32x4d']
        self.train_trans = 'train'
        self.tta_trans = 'tta'
        self.tta_num = 5
        self.val_trans = 'val'

        self.freeze_bn = False
        self.outputs = MODEL_PATH
        # 正则化
        self.l2_norm = 0.001
        # mix_up
        self.mix_up = None
        # label shuffle
        self.label_shuffle = False


def prn_obj(obj):
    print('\n'.join(['%s: %s' % item for item in obj.__dict__.items()]))


cfg = Config()
# print('参数列表如下')
# print(prn_obj(cfg))
