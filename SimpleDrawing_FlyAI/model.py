# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base

from path import MODEL_PATH
from SD_Dataset import load_from_json, draw_cv2
from config import cfg
import utils
from numpy_balance import get_balance_label
test_trans = utils.get_trans()['val']
tta_trans = utils.get_trans()['train']

# __import__('net', fromlist=["Net"])

TORCH_MODEL_NAME = "model.pkl"
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.net_path)
        x_data = self.data.predict_data(**data)
        x_data = torch.from_numpy(x_data)
        outputs = self.net(x_data)
        prediction = outputs.data.numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        # if cfg.model_names
        models_list = []
        for cnn_name in cfg.model_names:
            real_name = cnn_name+'.pkl'
            cnn = torch.load(os.path.join(MODEL_PATH, real_name))
            cnn.to(device)
            cnn.eval()
            models_list.append(cnn)
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            # 加载数据
            x_data = load_from_json(x_data[0])
            x_data = draw_cv2(x_data, cfg.img_size)
            # x_data = test_trans(img=x_data)

            # 维度变换 如果使用tta则可以不变换
            # x_data = x_data.unsqueeze(0)
            x_data = utils.get_tta(x_data, 5, tta_trans)

            # gpu
            x_data = x_data.float().to(device)
            '''use sig model
            outputs = cnn(x_data)
            # 先cpu,然后转numoy
            outputs = outputs.cpu()
            prediction = outputs.data.numpy()
            # 如果是tta 默认使用TTA 则需要求均值
            prediction = numpy.mean(prediction, axis=0)
            prediction = utils.np_softmax(prediction)
            # print(prediction, numpy.sum(prediction))
            '''
            # use mutil
            prediction = utils.get_merge_result(models_list, x_data)
            prediction = self.data.to_categorys(prediction)
            labels.append(prediction)

        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=TORCH_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network, os.path.join(path, name))
