# -*- coding: utf-8 -*-
import argparse
from flyai.dataset import Dataset
from keras.models import Model as Keras_Model
from keras.utils import to_categorical
from model import Model
from path import MODEL_PATH,DATA_PATH
import numpy as np
import os
from processor import img_size
NUM_CLASS = 10
KERAS_MODEL_NAME = "model.h5"
'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()
bs = args.BATCH
epochs = args.EPOCHS
'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)
dataset.image_aug(rotation_range=10,target_size=(224,224),horizontal_flip=True,zoom_range=0.1,brightness_range=[0.9,1.1])
'''
实现自己的网络机构
'''
#接下来就是考虑什么模型可以来做这个人脸预测了,测试vgg_face,senet50
from keras_vggface.vggface import VGGFace
from keras.layers import *
from lin import categorical_focal_loss
vgg_model = VGGFace(include_top=False,model='senet50',input_shape=(224, 224, 3),weights=None)
# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper
path = remote_helper.get_remote_date('https://www.flyai.com/m/rcmalli_vggface_tf_notop_senet50.h5')
vgg_model.load_weights(path)

x = vgg_model.output
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(rate=0.5)(x)
x = Dense(128,activation='relu')(x)
x = Dropout(rate=0.5)(x)
o = Dense(10,activation='softmax',name='prediction')(x)
seque = Keras_Model(inputs=vgg_model.input,outputs=o)
#学习率设置
from keras.optimizers import *
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
sgd = SGD(lr=0.0001)
seque.compile(loss=categorical_focal_loss(gamma=2., alpha=0.5), optimizer=sgd, metrics=['accuracy'])
# seque.compile(loss='categorical_crossentropy', optimizer=adam(lr=1e-4,decay=1e-6), metrics=['accuracy'])
# print(seque.summary())
'''
dataset.get_step() 获取数据的总迭代次数
'''
# class_weights = {0: 1, 1: 1.2515387669959643, 2: 2.321928094887362, 3: 2.643856189774725, 4: 3.0588936890535687, 5: 2.473931188332412, 6: 1.8365012677171206, 7: 3.0588936890535687, 8: 5.643856189774724, 9: 4.058893689053568}
min_loss = float('inf')
best_score = 0
'''数据增强'''
from keras_vggface.utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
data_aug = ImageDataGenerator( preprocessing_function=preprocess_input,
                                rotation_range=10,
                                # brightness_range=[0.9,1.1],
                                horizontal_flip=True,
                               )

'''获取数据'''
# x_val, y_val = dataset.get_all_validation_data()
train_x,train_y,valid_x,valid_y = dataset.get_all_data()
all_x = np.concatenate([train_x, valid_x])
all_y = np.concatenate([train_y, valid_y])
val_rate = 0.1
train_size = int(len(all_x)*(1-val_rate))
train_x = all_x[:train_size]
train_y = all_y[:train_size]
valid_x = all_x[train_size:]
valid_y = all_y[train_size:]
steps_per_epoch = int(train_x.shape[0] / bs)
print('name like:', train_x[0],train_y[0])
print('train len is %d, val len is %d, all step is %d'%(train_x.shape[0], valid_x.shape[0], steps_per_epoch))

from keras.preprocessing import image
from keras_vggface.utils import preprocess_input
def Dataloder(train_x, train_y, bs,transform=None):
    N = train_x.shape[0]
    while True:
        bs_samples = [np.random.randint(N-1) for _ in range(bs)]
        # 加载数据
        batch_x = []
        batch_y = []
        for i in bs_samples:
            img_path = os.path.join(DATA_PATH, train_x[i]['image_path'])
            img = image.load_img(img_path, target_size=img_size)
            x = image.img_to_array(img)
            batch_x.append(x)

            label_onehot = np.zeros((NUM_CLASS), dtype=int)
            index = train_y[i]['label']
            label_onehot[index] = 1
            batch_y.append(label_onehot)
        bs_train_x = np.squeeze(batch_x)
        bs_train_y = np.squeeze(batch_y)
        if transform!=None:
            bs_train_x = transform(bs_train_x)
        yield bs_train_x,bs_train_y
train_loder = Dataloder(train_x,train_y,bs=bs,transform=preprocess_input)
valid_loder = Dataloder(valid_x,valid_y,bs=bs,transform=preprocess_input)

'''训练模型'''
class_weights = {0: 1.6640964032688612, 1: 3.049825500442439, 2: 2.682918523352304, 3: 3.275479172200439, 4: 3.814749582228726, 5: 3.333777812765106, 6: 4.095747413061207, 7: 4.622623552696639, 8: 5.012444766366645, 9: 7.829353995137498}
lr_checkpoint = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode='auto')
md_checkpoin = ModelCheckpoint(os.path.join(MODEL_PATH,KERAS_MODEL_NAME),monitor='val_loss',mode='auto')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
result = seque.fit_generator(train_loder,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    callbacks=[lr_checkpoint,md_checkpoin],
                    validation_data=valid_loder,
                    validation_steps=len(valid_x)//bs,
                    class_weight=class_weights,
						    verbose=2,
                    shuffle=True)

# for step in range(dataset.get_step()):
#     x_train, y_train = dataset.next_train_batch()
#     # x_val, y_val = dataset.next_validation_batch()
#     y_train = to_categorical(y_train,num_classes=NUM_CLASS)
#     history = seque.fit_generator(data_aug.flow(x_train, y_train, batch_size=bs,seed=7),
#                         steps_per_epoch=1,epochs=1,
#                         verbose=2)
#
#     if (step+1)%(dataset.get_step()//epochs) == 0 or step == dataset.get_step()-1:
#         score = seque.evaluate(x_val, to_categorical(y_val,num_classes=10), verbose=0)
#         print((step+1)//(dataset.get_step()//epochs), "val_loss is:", score[0],"val_acc is:",score[1])
#
#         if score[1] > best_score:
#             min_loss = score[0]
#             best_score = score[1]
#             '''
#             保存模型
#             '''
#             model.save_model(seque, MODEL_PATH, overwrite=True)
#             print("step %d, best accuracy %g" % (step, best_score))
#         print(str(step + 1) + "/" + str(dataset.get_step()))