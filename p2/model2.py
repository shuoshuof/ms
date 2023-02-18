# -*- coding: utf-8 -*-
# @Time : 2023/2/18 23:04
# @Author : shuoshuo
# @File : train.py
# @Project : ms
from tensorflow import keras
from dataset import *
from sklearn.model_selection import train_test_split
import os
import time
class mymodel(object):
    def __init__(self,input_shape_1=(5,26),input_shape_2=(2),class_num=7):
        self.input_shape_1 = input_shape_1
        self.class_num =class_num
        self.input_shape_2 = input_shape_2
    def get_model(self):

        inputs_2 = keras.Input(shape=self.input_shape_2)
        x2 = keras.layers.Dense(8,activation='relu')(inputs_2)


        x2 = keras.layers.Dense(32,activation='relu')(x2)
        x2 = keras.layers.Dense(64,activation='relu')(x2)
        x2 = keras.layers.Dense(32,activation='relu')(x2)
        x2 = keras.layers.Dropout(0.2)(x2)
        outputs = keras.layers.Dense(self.class_num,activation='softmax')(x2)
        model = keras.Model(inputs_2,outputs)
        model.compile(loss=keras.losses.MeanAbsoluteError(),optimizer="adam")
        model.summary()
        return model

if __name__ == '__main__':
    x1_train, x1_test,x2_train,x2_test, y_train, y_test = get_data('../p1_2/C_data_new.xls')

    model = mymodel().get_model()

    save_path = './models_save/%s' % (time.strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(save_path)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, verbose=1)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    save_weights = keras.callbacks.ModelCheckpoint(save_path + "/model_{epoch:02d}_{val_loss:.4f}.h5",
                                                   save_best_only=True, monitor='val_loss')
    callbacks_list = [reduce_lr, early_stop, save_weights]
    model.fit(x2_train,y_train,
              epochs=400,
              batch_size=32,
              validation_data=(x2_test,y_test),
              callbacks = [reduce_lr, early_stop, save_weights])
