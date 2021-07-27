import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from path import MODEL_PATH
# from model import TORCH_MODEL_NAME
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from warm_up_sche import GradualWarmupScheduler
# from label_smooth import LabelSmoothingCrossEntropy,FocalLoss
# import platform
from flyai.utils.log_helper import train_log
import utils


class Trainer(object):
    def __init__(self, net,
                 train_dataloader, valid_dataloader,
                 criterion,
                 optimizer,
                 scheduler=None,
                 epochs=30,
                 name='test.pkl',
                 mixup=None):
        self.net = net
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.best_score = -1
        self.epoch = 0
        self.name = name
        self.max_epoch = epochs
        self.mixup = mixup
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_model(self, name):
        model_name = os.path.join(MODEL_PATH, name)
        torch.save(self.net, model_name)

    def mixup_data(self, x, y, alpha, device):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def compute_loss(self, logits, target):
        return self.criterion(logits, target)

    def train(self):
        net = self.net
        optimizer = self.optimizer
        device = self.device
        net.to(device)
        net.train()

        acc1s = []
        losses = []
        for i, (inputs, targets) in enumerate(self.train_dataloader):

            inputs, targets = inputs.to(device), targets.to(device)

            if self.mixup != None:
                inputs, targets_a, targets_b, lam = self.mixup_data(
                    inputs, targets, self.mixup, device)

                outputs = net(inputs)
                loss = self.mixup_criterion(
                    outputs, targets_a, targets_b, lam)
            else:
                outputs = net(inputs)
                loss = self.compute_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc1s = utils.accuracy(outputs, targets)

            acc1s.append((train_acc1s[0].item()))
            losses.append(loss.item())
        acc = np.mean(acc1s)
        loss = np.mean(losses)
        print("训练集的acc:  %g" % (acc))
        return acc, loss

    def validate(self):
        net = self.net
        device = self.device
        net.to(device)
        net.eval()
        acc1s = []
        losses = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.valid_dataloader):

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = self.compute_loss(outputs, targets)

                valid_acc1s = utils.accuracy(outputs, targets)

                acc1s.append((valid_acc1s[0].item()))
                losses.append(loss.item())

        acc = np.mean(acc1s)
        loss = np.mean(losses)
        print("验证集的acc:  %g" % (acc))

        if acc > self.best_score:
            self.best_score = acc
            print('Best epoch:', self.epoch)
            self.save_model(self.name)
        return acc, loss

    def train_epochs(self):
        while self.epoch < self.max_epoch:
            self.epoch += 1
            s_time = time.time()
            acc, loss = self.train()
            val_acc, val_loss = self.validate()
            if self.scheduler != None:
                self.scheduler.step(self.epoch)
            s_time = time.time()-s_time
            print("这次花了这么长时间:%f"%s_time)
            train_log(train_loss=loss, train_acc=acc, val_acc=val_acc, val_loss=val_loss)

        # if not os.path.exists(os.path.join(MODEL_PATH, TORCH_MODEL_NAME)):
        #     self.save_model(TORCH_MODEL_NAME)