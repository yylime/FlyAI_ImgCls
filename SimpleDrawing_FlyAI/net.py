# -*- coding: utf-8 -*
import torchvision.models as models
import torch.nn as nn
from config import cfg
def get_model(model_name):
    model_list = []
    for name in model_name:
        if name == 'resnet34':
            cnn = models.resnet34(pretrained=True)
            cnn.avgpool = nn.AdaptiveAvgPool2d(1)
            cnn.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512, cfg.num_class)
            )
            model_list.append(cnn)
        elif name=='densenet121':
            cnn = models.densenet121(pretrained=True)
            cnn.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(1024, cfg.num_class)
            )
            model_list.append(cnn)
    return model_list