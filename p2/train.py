# -*- coding: utf-8 -*-
# @Time : 2023/2/18 23:04
# @Author : shuoshuo
# @File : train.py
# @Project : ms
import tensorflow as tf
from tensorflow import keras
from dataset import *
import os
import time
import sklearn
class mymodel(object):
    def __init__(self,input_shape_1=(5,26),input_shape_2=(4),input_shape_3=(5,1),class_num=7):
        self.input_shape_1 = input_shape_1
        self.class_num =class_num
        self.input_shape_2 = input_shape_2
        self.input_shape_3 = input_shape_3

    def get_model(self):
        #输入1
        inputs_1 = keras.Input(shape=self.input_shape_1)
        # x1= keras.layers.LSTM(units=32,return_sequences=True)(inputs_1)
        # x1 = keras.layers.Dropout(0.2)(x1)
        x1 = keras.layers.LSTM(units=64, return_sequences=True)(inputs_1)
        x1 = keras.layers.Dropout(0.2)(x1)
        # x1= keras.layers.LSTM(units=32,return_sequences=True)(x1)
        # x1 = keras.layers.Dropout(0.2)(x1)
        x1= keras.layers.LSTM(units=32,return_sequences=False)(x1)
        x1 = keras.layers.Dropout(0.2)(x1)
        x1 = keras.layers.Dense(8, activation='tanh')(x1)

        #输入2
        inputs_2 = keras.Input(shape=self.input_shape_2)
        x2 = keras.layers.Dense(32,activation='tanh')(inputs_2)
        x2=  keras.layers.Dense(8,activation='tanh')(x2)

        #输入3
        inputs_3 = keras.Input(shape=self.input_shape_3)
        x3 = keras.layers.LSTM(units=32, return_sequences=True)(inputs_3)
        x3= keras.layers.LSTM(units=32, return_sequences=False)(x3)
        x3 = keras.layers.Dropout(0.2)(x3)
        x3 = keras.layers.Dense(8,activation='tanh')(x3)

        #合并
        x = keras.layers.concatenate([x1, x2,x3])
        x = keras.layers.Dense(32,activation='tanh')(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(self.class_num,'softmax')(x)
        model = keras.Model([inputs_1,inputs_2,inputs_3],outputs)
        model.summary()
        return model
def my_loss(y_true,y_pred):
    # sum = tf.reduce_sum(y_true,axis=1,keepdims=True)
    # y_true = y_true/sum
    # return ((y_true-y_pred)*100)**2
    return tf.abs(y_true-y_pred)
def R_squared(y, y_pred):
  residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
  total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
  r2 = tf.subtract(1.0, tf.divide (residual, total))
  return r2
if __name__ == '__main__':
    x1_train, x1_test,x2_train,x2_test,x3_train,x3_test,y_train, y_test= get_data('./c_data_new1 (2).xls')

    model = mymodel().get_model()
    keras.utils.plot_model(model,to_file='./model.png',show_shapes=True)

    save_path = './models_save/%s' % (time.strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(save_path)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.1, patience=80, verbose=1)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_mae', patience=160, verbose=1)
    save_weights = keras.callbacks.ModelCheckpoint(save_path + "/model_{epoch:02d}_{val_mae:.4f}.h5",
                                                   save_best_only=True, monitor='val_mae')
    callbacks_list = [reduce_lr, early_stop, save_weights]
    model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(1e-3),metrics=['mae',R_squared])
    model.fit([x1_train,x2_train,x3_train],y_train,
              epochs=600,
              batch_size=32,
              validation_data=([x1_test,x2_test,x3_test],y_test),
              callbacks = [reduce_lr, early_stop, save_weights])
