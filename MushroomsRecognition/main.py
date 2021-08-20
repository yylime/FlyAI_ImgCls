# -*- coding: utf-8 -*-
import argparse
import configparser
import os
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from config import cfg, prn_obj
from net import get_model
from path import MODEL_PATH
import pandas as pd
from MyDataset import MyDataset
import torch.optim as optim
import torch
import random
from imgt import trainner, utils, lgbmer

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
args = parser.parse_args()

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("MushroomsRecognition")
        self.deal_with_data()

    def deal_with_data(self):
        img_label = pd.read_csv('data/input/MushroomsRecognition/train.csv')
        imgs = list(img_label['image_path'])
        labels = list(img_label['label'])

        random.seed(0)
        idx = random.sample(range(len(imgs)), len(imgs))
        imgs = [imgs[i] for i in idx]
        labels = [labels[i] for i in idx]
        # lable shuffle
        if cfg.label_shuffle:
            imgs, labels = utils.label_shuffling(imgs, labels)

        # print(idx)
        assert len(imgs) == len(labels)
        N = len(imgs)
        print("Data length %d" % N)
        split = int(N * cfg.split)
        train_imgs, valid_imgs = imgs[split:], imgs[:split]
        train_labels, valid_labels = labels[split:], labels[:split]

        transforms = utils.get_trans(size=cfg.img_size)

        self.train_dst = MyDataset(train_imgs, train_labels, transforms[cfg.train_trans])
        self.valid_dst = MyDataset(valid_imgs, valid_labels, transforms[cfg.val_trans])
        # 得到均值方差
        # print(utils.get_mean_and_std(self.train_dst))

        self.train_loader = torch.utils.data.DataLoader(self.train_dst, batch_size=cfg.bs, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_dst, batch_size=cfg.bs, shuffle=False)

        ## lgbm-dataloader
        self.lgbm_train_dst = MyDataset(train_imgs, train_labels, transforms[cfg.val_trans])
        self.lgbm_valid_dst = MyDataset(valid_imgs, valid_labels, transforms[cfg.val_trans])
        # 得到均值方差
        # print(utils.get_mean_and_std(self.train_dst))

        self.lgbm_train_loader = torch.utils.data.DataLoader(self.lgbm_train_dst, batch_size=cfg.bs, shuffle=True)
        self.lgbm_valid_loader = torch.utils.data.DataLoader(self.lgbm_valid_dst, batch_size=cfg.bs, shuffle=False)

    def train(self):
        # 使用多模型融合
        models_list = get_model(cfg.model_names)
        t_cfg = trainner.Trainer
        for i, cur_cnn in enumerate(models_list):
            cnn = cur_cnn
            # 因为要保存model
            name = cfg.model_names[i] + '.pkl'
            cnn.to(device)
            # 训练数据
            print("训练中的模型 %s" % name)

            trainer = t_cfg(cnn,
                            self.train_loader,
                            self.valid_loader,
                            cfg)
            trainer.train_epochs(name)

    def lgbm_train(self):
        t_lgbm = lgbmer.Lgbmer
        models_list = get_model(cfg.model_names)
        for i, cur_cnn in enumerate(models_list):
            name = cfg.model_names[i] + '.pkl'
            lgbm_name = cfg.model_names[i] + '.txt'
            trainer = t_lgbm(name, self.lgbm_train_loader, self.lgbm_valid_loader, cfg)
            trainer.train(lgbm_name)


if __name__ == '__main__':
    #
    print("参数设置如下")
    prn_obj(cfg)
    # main
    main = Main()
    main.download_data()
    main.train()
    if cfg.lgbm:
        main.lgbm_train()
