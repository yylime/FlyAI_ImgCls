from torch import nn
from torchvision import models
import torch
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'
device = torch.device(device)

# 如果我想自己来构建这个se-net,我是不是只需要构建他的fc即可

cnn = models.resnet50(pretrained=True)
cnn.fc = nn.Linear(2048, 8)
cnn.to(device)

print(cnn)