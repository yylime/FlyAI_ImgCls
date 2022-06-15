# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
from flyai_sdk import FlyAI, DataHelper, DATA_PATH, MODEL_PATH
from config import cfg, prn_obj
from net import get_model
import pandas as pd
from MyDataset import MyDataset
import torch.optim as optim
import torch
import random
import trainner, utils


if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


# 所有标签列表
all_label_list = ['beach', 'circularfarmland', 'cloud', 'desert', 'forest', 'mountain',
                  'rectangularfarmland', 'residential', 'river', 'snowberg']

DATA_ID = "ReversoContextClass"

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


class Main(FlyAI):

    def download_data(self):
        data_helper = DataHelper()
        data_helper.download_from_ids(DATA_ID)
        self.deal_with_data()

    def deal_with_data(self):
        img_label = pd.read_csv(os.path.join(DATA_PATH, DATA_ID, 'train.csv'))
        imgs = list(img_label['name'])
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

        self.train_loader = torch.utils.data.DataLoader(self.train_dst, batch_size=cfg.bs, shuffle=False)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_dst, batch_size=cfg.bs, shuffle=False)


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


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.train()
