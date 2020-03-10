import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from path import MODEL_PATH
from model import Torch_MODEL_NAME
from torch.optim.lr_scheduler import ReduceLROnPlateau
from warm_up_sche import GradualWarmupScheduler
from label_smooth import LabelSmoothingCrossEntropy
import platform


class Trainer(object):
    def __init__(self, net, train_dataloader, valid_dataloader, epochs=30):
        self.net = net
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.best_score = 0.1
        self.train_score = 0.1
        self.epoch = 0
        self.max_epoch = epochs
        self.save_best = True
        self.valid_freq = 1
        self.run_valid = False
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.criterion = LabelSmoothingCrossEntropy()
        self.optimizer = torch.optim.SGD(self.net.parameters(), 0.001, momentum=0.9, weight_decay=5.0e-4, nesterov=True)
        self.scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=1e-6)
        # self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=3, verbose=True,
        #                                       eps=1e-6)
        self.scheduler_warmup = GradualWarmupScheduler(self.optimizer, multiplier=10, total_epoch=5,
                                                       after_scheduler=self.scheduler_cosine)

    def save_model(self, name):
        model_name = os.path.join(MODEL_PATH, name)
        torch.save(self.net, model_name)

    def compute_loss(self, logits, target):
        return self.criterion(logits, target)

    def train(self):
        net, optimizer = self.net, self.optimizer
        device = self.device

        preds = []
        targets = []
        loss_hist = []

        net.train()
        start_time = time.time()
        used_time = 0
        cpu_time = 0

        for X, target in self.train_dataloader:
            cpu_time += time.time() - start_time

            targets.extend(target.cpu().numpy().tolist())
            X, target = X.to(device), target.to(device)

            logits = net(X)
            loss = self.compute_loss(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(1).cpu().numpy().tolist()
            preds.extend(pred)
            loss_hist.append(loss.item())
            end_time = time.time()
            used_time += end_time - start_time
            start_time = time.time()

        targets = np.array(targets)
        preds = np.array(preds)

        acc = np.sum(targets == preds) / len(targets)
        loss = np.mean(loss_hist)

        print('Train Loss: {:.4f}, Time: {:.1f}, {:.1f},  Acc: {:.4f}'.format(
            loss, used_time, cpu_time, acc
        ))

    def validate(self):
        net = self.net
        device = self.device

        targets = []
        preds = []
        loss_hist = []

        net.eval()
        start_time = time.time()

        for X, target in self.valid_dataloader:
            targets.extend(target.cpu().numpy().tolist())
            X, target = X.to(device), target.to(device)

            with torch.no_grad():
                logits = net(X)
                loss = self.compute_loss(logits, target)
                pred = logits.argmax(1).cpu().numpy().tolist()
                preds.extend(pred)
                loss_hist.append(loss.item())

        end_time = time.time()
        targets = np.array(targets)
        preds = np.array(preds)
        acc = np.sum(targets == preds) / len(targets)
        loss = np.mean(loss_hist)

        print('Valid Loss: {:.4f}, Time: {:.1f}, Acc: {:.4f},'.format(
            loss, end_time - start_time, acc
        ))

        if self.save_best and acc > self.best_score:
            self.best_score = acc
            print('Best epoch:', self.epoch)
            self.save_model(Torch_MODEL_NAME)
        return loss

        # self.run_valid = self.run_valid | (self.best_score >= 0.89) | (self.epoch > int(0.7 * self.max_epoch))

    def train_epochs(self):
        while self.epoch < self.max_epoch:
            self.epoch += 1
            self.train()

            # if self.epoch % self.valid_freq == 0:
            loss = self.validate()
            # 学习率调整策略
            self.scheduler_warmup.step()

        if not os.path.exists(os.path.join(MODEL_PATH, Torch_MODEL_NAME)):
            self.save_model(Torch_MODEL_NAME)