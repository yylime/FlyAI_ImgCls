# -*- coding: utf-8 -*
import numpy
from PIL import Image
from flyai.processor.base import Base
from flyai.processor.download import check_download
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from path import DATA_PATH

# img_size = (256, 256)
test_size = (224, 224)
from torchvision.transforms import functional  as F

test_trans = transforms.Compose([
    transforms.Resize(test_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


class Processor(Base):
    def input_x(self, image_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        path = check_download(image_path, DATA_PATH)
        path = path.replace('\\', '/')
        image = Image.open(path).convert('RGB')
        return image

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, label):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        return label

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return numpy.argmax(data)
