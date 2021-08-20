# -*- coding: utf-8 -*
from flyai.framework import FlyAI
import numpy as np
import os
import torch
from path import MODEL_PATH, DATA_PATH
from config import cfg
from imgt import utils
import PIL.Image as Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
test_trans = utils.get_trans(size=cfg.img_size)[cfg.val_trans]
tta_trans = utils.get_trans(size=cfg.img_size)[cfg.tta_trans]

TORCH_MODEL_NAME = "model.pkl"
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
            real_name = cnn_name+'.pkl'
            name = os.path.join(MODEL_PATH, real_name)
            self.models_list.append(name)


    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"image_path":".\/data\/input\/FishClassification\/image\/0.png"}
        :return: 模型预测成功之后返回给系统样例 {"label":"0"}
        '''
        x_data = Image.open(image_path).convert("RGB")

        x_data = utils.get_tta(x_data, cfg.tta_num, tta_trans)
        # gpu
        x_data = x_data.float().to(device)

        # use mutil
        prediction = utils.get_merge_result(self.models_list, x_data, tta=True, use_lgbm=cfg.lgbm)

        prediction = np.argmax(prediction)
        return {"label": prediction}

if __name__ == '__main__':
    flyai_predictor = Prediction()
    flyai_predictor.load_model()
    p = flyai_predictor.predict('data/input/MushroomsRecognition/image/057_jOjOWsmPwSs.jpg')
    print(p)