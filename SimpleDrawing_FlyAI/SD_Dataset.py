#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/8 15:06
# @Author  : yulin
# @E-mail  : 844202100@qq.com 
# @School  : bupt
# @File    : SD_Dataset.py
import json
import os
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
from PIL import Image
import torch
import  matplotlib.pylab as plt
from path import DATA_PATH
import random


def load_from_json(json_path):
    path = os.path.join(DATA_PATH, json_path)
    with open(path) as f:
        draw = json.load(f)
    raw_strokes = draw['drawing']
    return raw_strokes


# def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
#     BASE_SIZE = 299
#     img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
#     for t, stroke in enumerate(raw_strokes):
#
#         str_len = len(stroke[0])
#         for i in range(len(stroke[0]) - 1):
#
#             # dot dropout
#             if np.random.uniform() > 0.95:
#                 continue
#
#             color = 255 - min(t, 10) * 13 if time_color else 255
#             _ = cv2.line(img, (stroke[0][i] + 22, stroke[1][i] + 22),
#                          (stroke[0][i + 1] + 22, stroke[1][i + 1] + 22), color, lw)
#
#     img = cv2.resize(img, (size, size))
#     rt_img = np.zeros((size, size, 3))
#     rt_img[:, :, 0] = img
#     rt_img[:, :, 1] = rt_img[:, :, 0]
#     rt_img[:, :, 2] = rt_img[:, :, 0]
#     rt_img = Image.fromarray(np.uint8(rt_img))
#     return rt_img
# 第四次测试
def draw_cv2(raw_strokes, size=256, lw=6):
    BASE_SIZE = 299
    img = np.zeros((BASE_SIZE, BASE_SIZE, 3), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        points_count = len(stroke[0]) - 1
        grad = 255 // points_count
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i]+22, stroke[1][i]+22), (stroke[0][i + 1]+22, stroke[1][i + 1]+22),
                         (255, 255 - min(t, 10) * 13, max(255 - grad * i, 20)), lw)

    img = cv2.resize(img, (size, size))
    rt_img = Image.fromarray(np.uint8(img))
    return rt_img


class SDDataset(Dataset):
    def __init__(self, img_path, img_label, img_size, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, index):
        raw_strokes = load_from_json(self.img_path[index])
        img = draw_cv2(raw_strokes, self.img_size)

        if self.transform is not None:
            img = self.transform(img)

        label = self.img_label[index]
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.img_label)


if __name__ == '__main__':
    pass
    path = 'draws/draw_239207.json'
    img = draw_cv2(load_from_json(path),size=64)
    print(np.array(img))
    print(img, type(img))
    plt.imshow(img)
    plt.show()
