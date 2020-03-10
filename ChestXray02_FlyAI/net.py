# -*- coding: utf-8 -*
import torchvision.models as models
import torch.nn as nn
from config import cfg
from cbam_resnet import resnet34_cbam
from efficientnet_pytorch import EfficientNet
import torch


def get_model(model_name):
    model_list = []
    for name in model_name:
        if name == 'resnet34':
            cnn = models.resnet34(pretrained=True)
            cnn.avgpool = nn.AdaptiveAvgPool2d(1)
            cnn.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(512, cfg.num_class)
            )
            model_list.append(cnn)
        if name == 'resnet34_cbam':
            cnn = resnet34_cbam(pretrained=True)
            cnn.avgpool = nn.AdaptiveAvgPool2d(1)
            cnn.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(512, cfg.num_class)
            )
            model_list.append(cnn)
        elif name=='densenet121':
            cnn = models.densenet121(pretrained=True)
            cnn.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1024, cfg.num_class)
            )
            model_list.append(cnn)
        elif name=='resnet50':
            cnn = models.resnet50(pretrained=True)
            cnn.avgpool = nn.AdaptiveAvgPool2d(1)
            cnn.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(2048, cfg.num_class)
            )
            model_list.append(cnn)
        elif name == 'efficientnet-b0':
            cnn = EfficientNet.from_pretrained('efficientnet-b0', num_classes=cfg.num_class)
            model_list.append(cnn)

    return model_list