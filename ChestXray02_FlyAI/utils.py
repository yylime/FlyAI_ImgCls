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


def get_trans(size=224):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return \
        {
            'train': torch_transforms.Compose([
                torch_transforms.Resize(size),
                torch_transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                torch_transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05),
                                              fillcolor=(0, 0, 0)),
                torch_transforms.ToTensor(),
                torch_transforms.Normalize(mean, std),

            ]),
            'val': torch_transforms.Compose([
                torch_transforms.Resize(size),
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
    # tta_trans = transforms.Compose([
    #     transforms.Resize(size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])
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
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
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
