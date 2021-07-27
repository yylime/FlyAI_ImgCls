#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/8 15:40
# @Author  : yulin
# @E-mail  : 844202100@qq.com 
# @School  : bupt
# @File    : config.py
class Config:
    def __init__(self):
        self.lr = 0.01
        self.img_size = 224
        self.bs = 65
        self.epochs = 42
        self.num_class = 20
        # self.model_names = ['resnet18']
        self.model_names = ['resnet34_cbam', 'densenet121', 'efficientnet-b0']


def prn_obj(obj):
    print('\n'.join(['%s: %s' % item for item in obj.__dict__.items()]))


cfg = Config()
# print('参数列表如下')
# print(prn_obj(cfg))