# -*- coding: utf-8 -*
import os
from flyai.model.base import Base
from keras.engine.saving import load_model
from keras_vggface.utils import preprocess_input
from path import MODEL_PATH
from lin import categorical_focal_loss

KERAS_MODEL_NAME = "model.h5"


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
        self.model_path = os.path.join(MODEL_PATH, KERAS_MODEL_NAME)
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path,custom_objects={'categorical_focal_loss_fixed':categorical_focal_loss(gamma=2., alpha=0.5)})

    '''
    评估一条数据
    '''

    def predict(self, **data):
        if self.model is None:
            self.model = load_model(self.model_path)
        data = self.dataset.predict_data(**data)
        data = preprocess_input(data)
        data = self.model.predict(data)
        data = self.dataset.to_categorys(data)
        return data

    '''
    评估的时候会调用该方法实现评估得分
    '''

    def predict_all(self, datas):
        if self.model is None:
            self.model = load_model(self.model_path)
        labels = []
        for data in datas:
            data = self.dataset.predict_data(**data)
            data = preprocess_input(data)
            data = self.model.predict(data)
            data = self.dataset.to_categorys(data)
            labels.append(data)

        return labels

    '''
    保存模型的方法
    '''

    def save_model(self, model, path, name=KERAS_MODEL_NAME, overwrite=False):
        super().save_model(model, path, name, overwrite)
        model.save(os.path.join(path, name))