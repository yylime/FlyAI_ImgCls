# -*- coding: utf-8 -*
from flyai.framework import FlyAI
import numpy as np
import os
import torch
from path import MODEL_PATH, DATA_PATH
from config import cfg
import utils
import PIL.Image as Image

test_trans = utils.get_trans(size=cfg.img_size)['val']
tta_trans = utils.get_trans(size=cfg.img_size)['tta']
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


class Prediction(FlyAI):
    def __init__(self):
        self.models_list = None

    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        self.models_list = []
        for cnn_name in cfg.model_names:
            real_name = cnn_name + '.pkl'
            cnn = torch.load(os.path.join(MODEL_PATH, real_name))
            cnn.to(device)
            cnn.eval()
            self.models_list.append(cnn)

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"image_path":".\/data\/input\/FishClassification\/image\/0.png"}
        :return: 模型预测成功之后返回给系统样例 {"label":"0"}
        '''
        xs_data = utils.crop_picture_for_prediction(image_path)
        labels = []
        predictions = []
        for x_data in xs_data:
            x_data = utils.get_tta(x_data, 5, tta_trans)
            # gpu
            x_data = x_data.float().to(device)

            # use mutil
            prediction = utils.get_merge_result(self.models_list, x_data)
            predictions.append(prediction)
            label = np.argmax(prediction)
            labels.append(str(int(label)))

        if labels[0] != '0':
            labels[0] = '0'
        if labels[1] not in ['0', '1']:
            if predictions[1][0] < predictions[1][1]:
                labels[1] = '1'
            else:
                labels[1] = '0'
        pred_label = ','.join(labels)
        return {"label": pred_label}

if __name__ == '__main__':
    flyai_predictor = Prediction()
    flyai_predictor.load_model()
    p = flyai_predictor.predict('data/input/WaterMeterNumberRecognition/image/2898.jpg')
    print(p)