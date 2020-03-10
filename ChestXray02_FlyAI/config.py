#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/8 15:40
# @Author  : yulin
# @E-mail  : 844202100@qq.com 
# @School  : bupt
# @File    : config.py
class Config:
    def __init__(self):
        self.lr = 3e-4
        self.img_size = 224
        self.bs = 64
        self.epochs = 25
        self.num_class = 4
        self.model_names = ['efficientnet-b0', 'densenet121','resnet34']


def prn_obj(obj):
    print('\n'.join(['%s: %s' % item for item in obj.__dict__.items()]))


cfg = Config()
# print('参数列表如下')
# print(prn_obj(cfg))
