# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse
import torch
from flyai.dataset import Dataset
from path import MODEL_PATH
from config import cfg, prn_obj
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
from SD_Dataset import SDDataset
import torchvision.models as models
import timm
from Trainer import Trainer
from net import get_model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


'''
项目的超参
'''

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
args = parser.parse_args()


'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
bs = args.BATCH
epochs = args.EPOCHS
if __name__ == '__main__':
    print(prn_obj(cfg))
    x_train, y_train, x_val, y_val = dataset.get_all_processor_data()
    # concatenate
    all_x = np.concatenate((x_train, x_val))
    all_y = np.concatenate((y_train, y_val))
    split = int(len(all_x) * 0.1)
    x_train, x_val = all_x[:-split], all_x[-split:]
    y_train, y_val = all_y[:-split], all_y[-split:]
    print(x_train[0], x_val[0])
    print("训练数据集长度：", len(x_train))
    print("测试数据集长度：", len(x_val))
    # 判断gpu是否可用
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # 构建数据读取


    transforms = utils.get_trans()

    train_dst = SDDataset(x_train, y_train, img_size=cfg.img_size, transform=transforms['train'])
    valid_dst = SDDataset(x_val, y_val, img_size=cfg.img_size, transform=transforms['val'])

    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=cfg.bs, shuffle=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dst, batch_size=cfg.bs, shuffle=False, pin_memory=True)

    # 得到均值方差
    # print(utils.get_mean_and_std(train_dst))
    # (tensor([0.0821, 0.0737, 0.0481]), tensor([0.2620, 0.2388, 0.1727]))
    # 加载模型

    # 必须使用该方法下载模型，然后加载
    from flyai.utils import remote_helper

    '''
    # path = remote_helper.get_remote_date('https://www.flyai.com/m/resnet34-333f7ec4.pth')
    # cnn.load_state_dict(torch.load(path))
    # cnn = models.resnet34(pretrained=True)
    # cnn.avgpool = nn.AdaptiveAvgPool2d(1)
    # cnn.fc = nn.Sequential(
    #     nn.Dropout(0.3),
    #     nn.Linear(512, cfg.num_class)
    # )
    # cnn = timm.create_model('seresnext26t_32x4d',pretrained=True, num_classes=40)
    # cnn.to(device)
    '''
    # 使用多模型融合
    models_list = get_model(cfg.model_names)
    for i, cur_cnn in enumerate(models_list):
        cnn = cur_cnn
        # 因为要保存model
        name = cfg.model_names[i]+'.pkl'
        cnn.to(device)
        # 训练数据
        loss_fn = nn.CrossEntropyLoss()
        # loss_fn = utils.LabelSmoothingCrossEntropy()
        optimizer = optim.Adam(cnn.parameters(), lr=cfg.lr, weight_decay=5e-4)
        # optimizer = optim.SGD(cnn.parameters(), lr=cfg.lr, momentum=0.9, nesterov=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.5)
        print("训练中的模型 %s" % name)

        trainer = Trainer(cnn,
                          train_loader, valid_loader,
                          loss_fn,
                          optimizer,
                          scheduler,
                          epochs=cfg.epochs,
                          name=name
                          )
        trainer.train_epochs()
