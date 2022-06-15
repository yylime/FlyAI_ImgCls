# -*- coding: utf-8 -*
from flyai_sdk import FlyAI, DATA_PATH, MODEL_PATH
import numpy as np
import os
import torch
from config import cfg
import PIL.Image as Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import utils
test_trans = utils.get_trans(size=cfg.img_size)[cfg.val_trans]
tta_trans = utils.get_trans(size=cfg.img_size)[cfg.tta_trans]

# 所有标签列表
all_label_list = ['beach', 'circularfarmland', 'cloud', 'desert', 'forest', 'mountain',
                  'rectangularfarmland', 'residential', 'river', 'snowberg']

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

class Prediction(FlyAI):

    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型，然后return中必须返回该模型
        '''
        self.models_list = []
        for cnn_name in cfg.model_names:
            real_name = cnn_name+'.pkl'
            name = os.path.join(MODEL_PATH, real_name)
            self.models_list.append(name)

    def predict(self, image_path):
        x_data = Image.open(image_path).convert("RGB")

        x_data = utils.get_tta(x_data, cfg.tta_num, tta_trans)
        # gpu
        x_data = x_data.float().to(device)

        # use mutil
        prediction = utils.get_merge_result(self.models_list, x_data, tta=True, use_lgbm=False)
        prediction = np.argmax(prediction)

        label = all_label_list[prediction]
        return {"label": label}

if __name__ == '__main__':
    prediction = Prediction()
    prediction.load_model()
    res = prediction.predict("/Users/yylime/Documents/repos/FlyAI_cls/data/input/ReversoContextClass/images/108116.jpg")
    print(res)
