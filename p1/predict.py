# -*- coding: utf-8 -*-
# @Time : 2023/2/18 12:40
# @Author : shuoshuo
# @File : predict.py
# @Project : ms
from tensorflow import keras
def show_data(x,y):
    pass

if __name__ == '__main__':
    model = keras.models.load_model('./models_save/2023_02_18_12_34_35/model_19_0.0934.h5')
    model.summary()