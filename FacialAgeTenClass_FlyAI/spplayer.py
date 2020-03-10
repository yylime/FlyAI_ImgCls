#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/16 10:19
# @Author  : lin
# @Site    : 
# @File    : spplayer.py


from keras.layers import Layer
import keras.backend as K
import math


class SPP(Layer):
    '''
    主要是想写一个金字塔的池化
    这里有个问题怎么得到上一层的输出的大小啊，可以直接调用吗？？？！！
    默认使用 tensorflow
    输入形式：(samples,cols,rows,channels)
    输出形式：(samples,num_channels*[i*i,for i in pool_list])

    '''

    def __init__(self,pooling_list,**kwargs):
        super(SPP, self).__init__(**kwargs)
        self.pooling_list = pooling_list
        self.num_channels = None
        self.single_channel_dims = sum([i*i for i in self.pooling_list])
        # inherit from Layer


    # build for create weights but pooling not weights
    def build(self, input_shape):
        self.num_channels = input_shape[3]

    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.num_channels*self.single_channel_dims)

    # for config
    def get_config(self):
        config = {'pooling_list': self.pooling_list}
        base_config = super(SPP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # do pooling
    def call(self, x, **kwargs):
        #make sure the inputs is a list?
        out_shape = K.shape(x)
        input_shape = K.int_shape(x)
        #note the input_shape = (samples,cols,rows,channels)
        num_rows = input_shape[1]
        num_cols = input_shape[2]
        #divided pooling_list
        row_length = [num_rows/i for i in self.pooling_list]
        col_length = [num_cols/i for i in self.pooling_list]

        outputs = []
        for i,pooling_size in enumerate(self.pooling_list):
            #这里需要注意了，看看原文
            h_wid = math.ceil(row_length[i])
            w_wid = math.ceil(col_length[i])
            h_str = math.floor(row_length[i])
            w_str = math.floor(col_length[i])
            out = K.pool2d(x=x,pool_size=(h_wid,w_wid),strides=(h_str,w_str))
            out = K.reshape(x=out,shape=(out_shape[0],-1))
            outputs.append(out)
        return K.concatenate(outputs,1)