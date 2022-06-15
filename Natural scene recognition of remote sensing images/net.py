# -*- coding: utf-8 -*
from email.policy import strict
from joblib import MemorizedResult
import torchvision.models as models
import torch.nn as nn
from config import cfg
from cbam_resnet import resnet50_cbam
from efficientnet_pytorch import EfficientNet
import torch
import timm
from torch import Tensor
import torch.functional as F
import os
from collections import OrderedDict

# 必须使用该方法下载模型，然后加载
from flyai_sdk import download_model
def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        if state_dict_key:
            state_dict = checkpoint[state_dict_key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        print("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

def get_model(model_name):
    model_list = []
    for name in model_name:
        if name == 'resnet34':
            cnn = models.resnet34(pretrained=False)
            path = download_model('https://www.flyai.com/m/resnet34-333f7ec4.pth')
            cnn.load_state_dict(torch.load(path))
            cnn.avgpool = nn.AdaptiveAvgPool2d(1)
            cnn.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(512, cfg.num_class)
            )
            model_list.append(cnn)
        if name == 'resnet18':
            cnn = models.resnet18(pretrained=True)
            cnn.avgpool = nn.AdaptiveAvgPool2d(1)
            cnn.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(512, cfg.num_class)
            )
            model_list.append(cnn)
        if name == 'resnet50_cbam':
            cnn = resnet50_cbam(pretrained=False)
            # 必须使用该方法下载模型，然后加载
            path = download_model('https://www.flyai.com/m/resnet50-19c8e357.pth')
            cnn.load_state_dict(torch.load(path), strict=False)
            cnn.avgpool = nn.AdaptiveAvgPool2d(1)
            cnn.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(2048, cfg.num_class)
            )
            model_list.append(cnn)
        elif name == 'densenet121':
            cnn = models.densenet121(pretrained=False)
            # 必须使用该方法下载模型，然后加载
            path = download_model('https://www.flyai.com/m/densenet121-a639ec97.pth')
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
            path = download_model('https://www.flyai.com/m/wide_resnet50_2-95faca4d.pth')
            cnn = models.wide_resnet50_2(pretrained=False)
            cnn.load_state_dict(torch.load(path))
            cnn.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(2048, cfg.num_class)
            )
            model_list.append(cnn)
        elif name == 'efficientnet-b0':
            path = download_model('https://www.flyai.com/m/efficientnet-b0-355c32eb.pth')
            cnn = EfficientNet.from_pretrained('efficientnet-b0', weights_path=path, num_classes=cfg.num_class)
            model_list.append(cnn)

        elif name == 'efficientnet-b3':
            path = download_model('https://www.flyai.com/m/efficientnet-b3-5fb5a3c3.pth')
            cnn = EfficientNet.from_pretrained('efficientnet-b3', weights_path=path, num_classes=cfg.num_class)
            model_list.append(cnn)

        elif name == 'efficientnet-b4':
            path = download_model('https://www.flyai.com/m/efficientnet-b4-6ed6700e.pth')
            cnn = EfficientNet.from_pretrained('efficientnet-b4', weights_path=path, num_classes=cfg.num_class)
            model_list.append(cnn)

        elif name == 'densenet169':
            cnn = models.densenet169(pretrained=False)
            path = download_model('https://www.flyai.com/m/densenet169-f470b90a4.pth')
            cnn.load_state_dict(torch.load(path), strict=False)
            cnn.classifier = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(1664, cfg.num_class)
            )
            model_list.append(cnn)
        elif name == 'resnext50_32x4d':
            # 必须使用该方法下载模型，然后加载
            path = download_model('https://www.flyai.com/m/resnext50_32x4d-7cdf4587.pth')
            cnn = models.resnext50_32x4d(pretrained=False)
            cnn.load_state_dict(torch.load(path))
            cnn.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(2048, cfg.num_class)
            )
            model_list.append(cnn)

        elif name == "convnext_base":
            cnn = models.convnext_base(pretrained=True)
            cnn.classifier = nn.Sequential(
                LayerNorm2d((1024,), eps=1e-6, elementwise_affine=True),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(in_features=1024, out_features=cfg.num_class, bias=True)
            )
            model_list.append(cnn)

        elif name == 'xcit_small_24_p8_224_dist':
            # 必须使用该方法下载模型，然后加载
            # path = download_model('https://www.flyai.com/m/swin_large_patch4_window7_224_22k.pth')
            path = '/Users/yylime/.cache/torch/hub/checkpoints/xcit_small_24_p8_224_dist.pth'

            cnn = timm.create_model('xcit_small_24_p8_224_dist', pretrained=False)
            # static_dict = torch.load(path, map_location='cpu')['model']
            static_dict = load_state_dict(path)
            cnn.load_state_dict(static_dict, strict=False)
            
            model_list.append(cnn)
        elif name == 'resnet200d':
            # 必须使用该方法下载模型，然后加载
            path = download_model('https://pytorch-weights.bj.bcebos.com/resnet200d_ra2-bdba9bf9.pth')
            # path = '/Users/yylime/.cache/torch/hub/checkpoints/resnet200d_ra2-bdba9bf9.pth'
            cnn = timm.create_model('resnet200d', pretrained=False)
            static_dict = torch.load(path)
            cnn.load_state_dict(static_dict)
            model_list.append(cnn)


    return model_list


if __name__ == "__main__":
    from pprint import pprint

    # model_names = timm.list_models(pretrained=True)
    # pprint(model_names)
    # model = timm.create_model('xcit_small_24_p8_224_dist', pretrained=True)
    # print(model)
    model = get_model(['resnet50_cbam'])
    print(model[0])
