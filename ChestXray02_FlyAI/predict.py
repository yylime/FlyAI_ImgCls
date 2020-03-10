#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/21 15:41
# @Author  : yulin
# @E-mail  : 844202100@qq.com 
# @School  : bupt
# @File    : predict.py
from flyai.dataset import Dataset
from model import Model

class_mapping = {0: "1", 1:  "2", 2: "3", 3: "4"}

data = Dataset()
model = Model(data)
x_test = [{'image_path': 'images/00002783_000.png'}, {'image_path': 'images/00002783_000.png'}]
p = model.predict_all(x_test)
print(p)