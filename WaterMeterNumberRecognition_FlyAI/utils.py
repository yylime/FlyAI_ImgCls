#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/8 15:52
# @Author  : yulin
# @E-mail  : 844202100@qq.com 
# @School  : bupt
# @File    : utils.py
import torch
import torchvision.transforms as torch_transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
import os
import sys
import cv2

DATA_PATH = 'data/input/WaterMeterNumberRecognition/'
CROP_PATH = 'data/input/images/'

C_PATH = os.path.join(sys.path[0], 'data', 'input', 'images')
if not os.path.exists(C_PATH):
    os.makedirs(C_PATH)


def crop_picture_for_prediction(path):
    image = cv2.imread(path)

    # 去噪声
    # image = cv2.fastNlMeansDenoising(image, None, 16, 10, 7)

    height, width, c = image.shape
    cropped_images = []

    for left in range(5):
        region = image[0:height, round(left * width / 5):round((left + 1) * width / 5)]
        cropped_images.append(Image.fromarray(region))

    return cropped_images


def crop_save_images(path, label):
    path = os.path.join(DATA_PATH, path)
    image_idx = path.split('/')[-1].split('.')[0]
    cropped_images = crop_picture_for_prediction(path)
    label = label.split(',')

    for left, (img, l) in enumerate(zip(cropped_images, label)):
        region_path = os.path.join(CROP_PATH, image_idx + '_' + str(left) + '_' + l + '.jpg')
        img.save(region_path)
    return None


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class whResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


def get_trans(size=224):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return \
        {
            'train': torch_transforms.Compose([
                torch_transforms.Resize((size, size)),
                # torch_transforms.RandomHorizontalFlip(),
                # torch_transforms.RandomVerticalFlip(),
                torch_transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                torch_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                torch_transforms.ToTensor(),
                torch_transforms.Normalize(mean, std),
                torch_transforms.RandomErasing()
            ]),
            'tta': torch_transforms.Compose([
                torch_transforms.Resize((size, size)),
                # torch_transforms.RandomHorizontalFlip(),
                # torch_transforms.RandomVerticalFlip(),
                torch_transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                torch_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                torch_transforms.ToTensor(),
                torch_transforms.Normalize(mean, std),
            ]),
            'val': torch_transforms.Compose([
                torch_transforms.Resize((size, size)),
                torch_transforms.ToTensor(),
                torch_transforms.Normalize(mean, std)
            ]),
        }


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def lin_comb(a, b, epsilon):
    return epsilon * a + b * (1 - epsilon)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon, self.reduction = epsilon, reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return lin_comb(loss / c, nll, self.epsilon)


def get_tta(img, n, tta_trans):
    imgs = []
    for _ in range(n):
        imgs.append(tta_trans(img=img))
    data = torch.stack(imgs)
    return data


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def np_softmax(x):
    assert len(x.shape) == 1
    x = np.exp(x)
    z = np.sum(x)
    return x / z


def get_merge_result(models, x_data, tta=True):
    preds = None
    for i, cnn in enumerate(models):
        outputs = cnn(x_data)
        # 先cpu,然后转numoy
        outputs = outputs.cpu()
        prediction = outputs.data.numpy()
        # 如果是tta 默认使用TTA 则需要求均值
        if tta:
            prediction = np.mean(prediction, axis=0)
        # 转化为概率
        prediction = np_softmax(prediction)
        if preds is None:
            preds = prediction
        else:
            preds += prediction
    return preds