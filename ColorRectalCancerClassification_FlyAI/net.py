## build CNN
# from torch import nn
# import torch
# from torchvision import models
# from flyai.utils import remote_helper
# path = remote_helper.get_remote_date('https://www.flyai.com/m/resnet18-5c106cde.pth')
# # path_densenet201 = remote_helper.get_remote_date('https://www.flyai.com/m/densenet201-c1103571.pth')
#
#
#
# class resnet18(nn.Module):
#     def __init__(self):
#         super(resnet18, self).__init__()
#         self.resnet = models.resnet18(pretrained=False)
#         # #加载参数
#         self.resnet.load_state_dict(torch.load(path))
#         self.resnet.fc = nn.Linear(512,8)
#
#
#     def forward(self, input):
#         x = self.resnet(input)
#         return x