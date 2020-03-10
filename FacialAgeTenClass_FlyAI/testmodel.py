#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/26 11:36
# @Author  : lin
# @Site    : 
# @File    : testmodel.py

from keras.applications import InceptionV3
base_model = InceptionV3(include_top=False)
for layer in base_model.layers:
    layer.trainable = False
print(base_model.summary())