#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/8 15:40
# @Author  : yulin
# @E-mail  : 844202100@qq.com 
# @School  : bupt
# @File    : config.py
from path import MODEL_PATH


class Config:
    def __init__(self):
        self.lr = 0.0001
        self.img_size = 384
        self.bs = 8
        self.epochs = 30
        self.num_class = 9
        self.split = 0.1
        self.model_names = ['cait_s24_384', 'efficientnet-b3', 'wide_resnet50', 'resnext50_32x4d']
        # self.model_names = ['wide_resnet50']
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

        # lgbm params
        self.lgbm = False
        self.lgbm_params = {'boosting_type': 'gbdt',
                            'n_estimators': 3000,
                            'objective': 'multiclass',
                            'num_class': 9,
                            'learning_rate': 0.01,
                            #           'bagging_fraction':0.8,
                            #           'class_weight':'balanced',
                            'early_stopping_rounds': 200,
                            'metric': {'multi_logloss'},
                            #           'reg_alpha' : 0.01,
                            #         'reg_lambda' :0.1,
                            }


def prn_obj(obj):
    print('\n'.join(['%s: %s' % item for item in obj.__dict__.items()]))


cfg = Config()
# print('参数列表如下')
# print(prn_obj(cfg))
