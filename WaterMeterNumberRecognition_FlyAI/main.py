# -*- coding: utf-8 -*-
import argparse
import os

from flyai.data_helper import DataHelper
from flyai.framework import FlyAI

import utils
from Trainer import Trainer
from config import cfg
from net import get_model
from path import MODEL_PATH
import pandas as pd
from MyDataset import MyDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import random
from utils import crop_save_images
from warm_up_sche import GradualWarmupScheduler

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)
CROP_PATH = 'data/input/images/'


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("WaterMeterNumberRecognition")

    def deal_with_data(self):
        img_label = pd.read_csv('data/input/WaterMeterNumberRecognition/train.csv')
        images = list(img_label['image_path'])
        labels = list(img_label['label'])

        # 预处理下图片不然太慢了
        print('图片预处理中')
        for img, label in zip(images, labels):
            crop_save_images(img, label)
        print('图片预处理结束')
        # redefine images and labels
        images = os.listdir(CROP_PATH)
        labels = [p.split('_')[-1].split('.')[0] for p in images]
        # analysis the number of label
        counts = [0]*20
        for i in labels:
            counts[int(i)] += 1
        print("The number of LABEL: ")
        print(counts)

        random.seed(233)
        idx = random.sample(range(len(images)), len(images))
        imgs = [images[i] for i in idx]
        labels = [labels[i] for i in idx]
        # print(idx)
        assert len(imgs) == len(labels)
        N = len(imgs)
        print("Data length %d" % N)
        split = int(N * 0.01)
        train_imgs, valid_imgs = imgs[split:], imgs[:split]
        train_labels, valid_labels = labels[split:], labels[:split]

        transforms = utils.get_trans(size=cfg.img_size)

        self.train_dst = MyDataset(train_imgs, train_labels, transforms['train'])
        self.valid_dst = MyDataset(train_imgs, train_labels, transforms['val'])
        # 得到均值方差
        # print(utils.get_mean_and_std(self.train_dst))

        self.train_loader = torch.utils.data.DataLoader(self.train_dst, batch_size=cfg.bs, shuffle=True,
                                                        pin_memory=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_dst, batch_size=cfg.bs, shuffle=False,
                                                        pin_memory=True)

    def train(self):
        # 使用多模型融合
        models_list = get_model(cfg.model_names)
        for i, cur_cnn in enumerate(models_list):
            cnn = cur_cnn
            # 因为要保存model
            name = cfg.model_names[i] + '.pkl'
            cnn.to(device)
            # 训练数据
            # loss_fn = nn.CrossEntropyLoss()
            loss_fn = utils.LabelSmoothingCrossEntropy()
            # optimizer = optim.Adam(cnn.parameters(), lr=cfg.lr, weight_decay=1e-4)
            optimizer = optim.SGD(cnn.parameters(), lr=cfg.lr/5, momentum=0.9, nesterov=True)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True)
            scheduler = GradualWarmupScheduler(optimizer, 5, cfg.epochs)
            print("训练中的模型 %s" % name)

            trainer = Trainer(cnn,
                              self.train_loader, self.valid_loader,
                              loss_fn,
                              optimizer,
                              scheduler,
                              epochs=cfg.epochs,
                              name=name
                              )
            trainer.train_epochs()


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train()