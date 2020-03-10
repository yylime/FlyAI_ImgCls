# -*- coding: utf-8 -*
import argparse
import torch
import torch.nn as nn
from flyai.dataset import Dataset
from torch.autograd import Variable
from model import Model
from path import MODEL_PATH
from dataloader import Dataloader
import os
from torchvision import models
import random
import numpy as np

# 数据获取辅助类
dataset = Dataset()

# 模型操作辅助类
model = Model(dataset)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
args = parser.parse_args()

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'
device = torch.device(device)

# 训练并评估模型
# img_size = (256, 256)
crop_size = (448, 448)
mini_size = (224, 224)
# 简单的数据增强
from torchvision import transforms
mini_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(mini_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing()
    ]),
    'val': transforms.Compose([
        transforms.Resize(mini_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing()
    ]),
    'val': transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据
dataset = Dataset()
x_train, y_train, x_val, y_val = dataset.get_all_data()

# 打乱数据
all_x = x_train + x_val
all_y = y_train + y_val
length = len(all_x)
split = int(length * 0.1)
random.seed(0)
samples = random.sample(range(length), length)

# list无法转换，使用numpy
all_x, all_y = np.array(all_x),np.array(all_y)
all_x = all_x[samples]
all_y = all_y[samples]
x_train, y_train, x_val, y_val = all_x[:-split], all_y[:-split], all_x[-split:], all_y[-split:]

bs = args.BATCH
epochs = args.EPOCHS
max_iteration = (len(x_train) // bs + 1) * epochs

# 打印数据统计
print('Train data: {:d}, Val data: {:d}'.format(len(x_train), len(x_val)))
print('Max iteration:', max_iteration)
print('Epochs:', epochs)
print('Batch size:', bs)

# build dataset
train_dataset = Dataloader(x_train, y_train, data_transforms['train'])
valid_dataset = Dataloader(x_val, y_val, data_transforms['val'])
# build mini dataset
mini_train = Dataloader(x_train, y_train, mini_data_transforms['train'])
mini_valid = Dataloader(x_val, y_val, mini_data_transforms['val'])

# densenet201
# cnn = models.densenet201(pretrained=True)
# cnn.classifier = nn.Linear(1920, 8)
# resnet50
cnn = models.wide_resnet50_2(pretrained=True)
cnn.fc = nn.Linear(2048, 8)
cnn.to(device)


from torch.utils.data import DataLoader as torch_DataLoader
from sampler import ImbalancedDatasetSampler

# define training and validation data loaders
train_dataloader = torch_DataLoader(train_dataset, batch_size=bs, sampler=ImbalancedDatasetSampler(train_dataset),
                                    shuffle=False)
valid_dataloader = torch_DataLoader(valid_dataset, batch_size=bs, shuffle=False)

# define mini train and valid data loaders
mini_train_dataloader = torch_DataLoader(mini_train, batch_size=bs, sampler=ImbalancedDatasetSampler(train_dataset),
                                    shuffle=False)
mini_valid_dataloader = torch_DataLoader(mini_valid, batch_size=bs, shuffle=False)

from Trainer import Trainer
# start mini_train for five epochs
print('开始使用小图片训练')
mini_trainer = Trainer(cnn, mini_train_dataloader, mini_valid_dataloader, epochs=15)
mini_trainer.train_epochs()
# real start
print("开始使用标准大小图片训练")
trainer = Trainer(cnn, train_dataloader, valid_dataloader, epochs=epochs)
trainer.train_epochs()