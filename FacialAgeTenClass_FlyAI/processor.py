# -*- coding: utf-8 -*
import numpy
from PIL import Image
from flyai.processor.base import Base
from flyai.processor.download import check_download
from path import DATA_PATH
from keras.preprocessing import image

img_size = (224,224)
class Processor(Base):
    def input_x(self, image_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''

        path = check_download(image_path, DATA_PATH)
        path = path.replace('\\','/')
        img = image.load_img(path, target_size=img_size)
        x = image.img_to_array(img)
        # image = Image.open(path)
        # image = image.resize(img_size)
        # x_data = numpy.array(image)
        # x_data = x_data.astype(numpy.float32)
        # x_data = numpy.multiply(x_data, 1.0 / 255.0)  ## scale to [0,1] from [0,255]
        #使用Keras不需要转置
        # x_data = numpy.transpose(x_data, (2, 0, 1))  ## reshape
        return x

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