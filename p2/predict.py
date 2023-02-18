# -*- coding: utf-8 -*-
# @Time : 2023/2/18 23:57
# @Author : shuoshuo
# @File : predict.py
# @Project : ms
import numpy as np
from tensorflow import keras
from dataset import *
import pandas as pd
if __name__ == '__main__':
  model = keras.models.load_model('./models_save/2023_02_19_02_14_44/model_81_0.0286.h5')
  model.summary()
  x1_train, x1_test,x2_train,x2_test, y_train, y_test = get_data('../p1_2/C_data_new.xls')
  y_pred = model.predict([x1_test,x2_test])
  # y_pred = np.round(y_pred,3)
  results = np.concatenate([y_test,y_pred],axis=1)
  results=np.round(results,2)
  print(results.shape)

  data = pd.DataFrame(results,columns=[ f'{i}' for i in range(1,8)]+[ f'pred {i}' for i in range(1,8)])
  data.to_csv('./pred_results.csv')