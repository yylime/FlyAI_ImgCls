# A Baseline for img-classification
# created by xiaolinzi
# time: 2021-07-09 17:04

import time
import os
import numpy as np
import torch
from torch import optim, nn
from .utils import accuracy, GradualWarmupScheduler, LabelSmoothingCrossEntropy


def mix_up_data(x, y, alpha, device):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# 冻结bn，在训练的时候不变化
def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class Trainer(object):
    def __init__(self, net, train_dataloader, valid_dataloader, cfg):
        self.net = net
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.net.to(self.device)

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.outputs = cfg.outputs
        self.max_epoch = cfg.epochs
        self.mix_up = cfg.mix_up
        self.freeze_bn = cfg.freeze_bn
        self.loss_fn = LabelSmoothingCrossEntropy()
        # 定义默认的optimizer以及scheduler
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=cfg.lr, momentum=0.9,
                                   nesterov=True, weight_decay=cfg.l2_norm)
        after_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs - 5, eta_min=1e-8)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 10, total_epoch=5,
                                                after_scheduler=after_scheduler)

    def save_model(self, name):
        name = os.path.join(self.outputs, name)
        torch.save(self.net, name)

    def mix_up_loss(self, pred, y_a, y_b, lam):
        return lam * self.loss_fn(pred, y_a) + (1 - lam) * self.loss_fn(pred, y_b)

    def compute_loss(self, logits, target):
        return self.loss_fn(logits, target)

    def train(self):
        self.net.train()
        if self.freeze_bn:
            self.net.apply(freeze_bn)
        acc1s, losses = [], []
        # 训练流程
        for i, (inputs, targets) in enumerate(self.train_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # 是否使用mix_up
            if self.mix_up is not None:
                inputs, targets_a, targets_b, lam = mix_up_data(
                    inputs, targets, self.mix_up, self.device)
                outputs = self.net(inputs)
                loss = self.mix_up_loss(
                    outputs, targets_a, targets_b, lam)
            else:
                outputs = self.net(inputs)
                loss = self.compute_loss(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录指标
            train_acc1s = accuracy(outputs, targets)
            acc1s.append((train_acc1s[0].item()))
            losses.append(loss.item())
        acc = np.mean(acc1s)
        loss = np.mean(losses)
        return acc, loss

    def validate(self):
        self.net.eval()
        acc1s, losses = [], []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.valid_dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.compute_loss(outputs, targets)
                # 记录z指标
                valid_acc1s = accuracy(outputs, targets)
                acc1s.append((valid_acc1s[0].item()))
                losses.append(loss.item())
        acc = np.mean(acc1s)
        loss = np.mean(losses)
        return acc, loss

    def train_epochs(self, path):
        best_score = -1
        for epoch in range(self.max_epoch):

            s_time = time.time()
            acc, loss = self.train()
            val_acc, val_loss = self.validate()
            if self.scheduler is not None:
                self.scheduler.step(epoch)
            s_time = time.time() - s_time

            # 保存模型
            if val_acc > best_score:
                best_score = val_acc
                self.save_model(path)

            # 打印结果
            print("Epoch %d 花了这么长时间:%g, 训练集的准确率：%g， 验证集的准确率:%g" %
                  (epoch, s_time, acc, val_acc))
