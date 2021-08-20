# -*- coding: utf-8 -*
import torchvision.models as models
import torch.nn as nn
from config import cfg
from cbam_resnet import resnet34_cbam
from efficientnet_pytorch import EfficientNet
import torch
import timm

# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper


def get_model(model_name):
    model_list = []
    for name in model_name:
        if name == 'resnet34':
            cnn = models.resnet34(pretrained=False)
            path = remote_helper.get_remote_date('https://www.flyai.com/m/resnet34-333f7ec4.pth')
            cnn.load_state_dict(torch.load(path))
            cnn.avgpool = nn.AdaptiveAvgPool2d(1)
            cnn.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(512, cfg.num_class)
            )
            model_list.append(cnn)
        if name == 'resnet18':
            cnn = models.resnet18(pretrained=False)
            # 必须使用该方法下载模型，然后加载
            path = remote_helper.get_remote_date('https://www.flyai.com/m/resnet18-5c106cde.pth')
            cnn.load_state_dict(torch.load(path))
            cnn.avgpool = nn.AdaptiveAvgPool2d(1)
            cnn.fc = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(512, cfg.num_class)
            )
            model_list.append(cnn)
        if name == 'resnet34_cbam':
            cnn = resnet34_cbam(pretrained=False)
            path = remote_helper.get_remote_date('https://www.flyai.com/m/resnet34-333f7ec4.pth')
            cnn.load_state_dict(torch.load(path), strict=False)
            cnn.avgpool = nn.AdaptiveAvgPool2d(1)
            cnn.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512, cfg.num_class)
            )
            model_list.append(cnn)
        elif name == 'densenet121':
            cnn = models.densenet121(pretrained=False)
            # 必须使用该方法下载模型，然后加载
            path = remote_helper.get_remote_date('https://www.flyai.com/m/densenet121-a639ec97.pth')
            cnn.load_state_dict(torch.load(path), strict=False)
            cnn.classifier = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(1024, cfg.num_class)
            )
            model_list.append(cnn)
        elif name == 'resnet50':
            cnn = models.resnet50(pretrained=True)
            cnn.avgpool = nn.AdaptiveAvgPool2d(1)
            cnn.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(2048, cfg.num_class)
            )
            model_list.append(cnn)
        elif name == 'wide_resnet50':
            path = remote_helper.get_remote_data('https://www.flyai.com/m/wide_resnet50_2-95faca4d.pth')
            cnn = models.wide_resnet50_2(pretrained=False)
            cnn.load_state_dict(torch.load(path))
            cnn.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(2048, cfg.num_class)
            )
            model_list.append(cnn)
        elif name == 'efficientnet-b0':
            path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet-b0-355c32eb.pth')
            cnn = EfficientNet.from_pretrained('efficientnet-b0', weights_path=path, num_classes=cfg.num_class)
            model_list.append(cnn)

        elif name == 'efficientnet-b3':
            path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet-b3-5fb5a3c3.pth')
            cnn = EfficientNet.from_pretrained('efficientnet-b3', weights_path=path, num_classes=cfg.num_class)
            model_list.append(cnn)

        elif name == 'efficientnet-b4':
            path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet-b4-6ed6700e.pth')
            cnn = EfficientNet.from_pretrained('efficientnet-b4', weights_path=path, num_classes=cfg.num_class)
            model_list.append(cnn)

        elif name == 'densenet169':
            cnn = models.densenet169(pretrained=False)
            path = remote_helper.get_remote_date('https://www.flyai.com/m/densenet169-f470b90a4.pth')
            cnn.load_state_dict(torch.load(path), strict=False)
            cnn.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Dropout(0.2),
                nn.Linear(1024, cfg.num_class)
            )
            model_list.append(cnn)
        elif name == 'resnext50_32x4d':
            path = remote_helper.get_remote_date('https://www.flyai.com/m/resnext50_32x4d-7cdf4587.pth')
            cnn = models.resnext50_32x4d(pretrained=False)
            cnn.load_state_dict(torch.load(path))
            cnn.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(2048, cfg.num_class)
            )
            model_list.append(cnn)

        elif name == 'cait_s24_384':
            cnn = timm.create_model('cait_s24_384', pretrained=True, num_classes=cfg.num_class)
            model_list.append(cnn)

    return model_list


if __name__ == "__main__":
    from pprint import pprint

    model_names = timm.list_models(pretrained=True)
    pprint(model_names)
    model = timm.create_model('cait_s24_384', pretrained=False, num_classes=9)
    print(model)
    # model = get_model(['densenet169'])
    # print(model[0])
