# -*- coding: utf-8 -*-
# @Time : 2023/2/18 23:57
# @Author : shuoshuo
# @File : predict.py
# @Project : ms
import numpy as np
import tensorflow as tf
from tensorflow import keras
from dataset import *
import pandas as pd
def decode(y):
  y = y/np.sum(y,axis=1,keepdims=True)

  return y
def me(y_true,y_pred):
  bs,l = y_true.shape
  error =np.abs(y_true - y_pred)*100
  print(error)
  sum = np.sum(error)/(bs*l)
  print(sum)

def get_words_codes(word):
  # %%
  dict = {}
  for i in range(26):
    dict[chr(97 + i)] = i
  # print(dict)

  # %%
  words = word
  words_codes = []
  for word in words:
    codes = []
    for letter in word:
      code = np.zeros((26))
      # print(letter)
      code[dict[letter]] = 1
      codes.append(code)

    assert len(codes) == 5
    words_codes.append(codes)
    # print(words_codes)
  words_codes = np.array(words_codes)
  return words_codes
if __name__ == '__main__':
  model = keras.models.load_model('./models_save/2023_02_19_23_38_10/model_86_0.0256.h5',compile=False)
  model.summary()
  word = ['eerie']
  words_codes = get_words_codes(word)
  print(words_codes)

  x1 = words_codes
  x2 = np.array([[0.963021588518739,1,0.124130659586033,(620-202)/420]])
  x3 = np.array([[0.00430964271725761,0.041914331703822,0.026872502722745,0.0174762306610697,0.03795404219599 ]])
  y = model.predict([x1,x2,x3])
  print(y)
  avg = np.multiply(y,np.arange(1,8))
  avg = np.sum(avg,axis=1)
  print(avg)
  # x1_train, x1_test,x2_train,x2_test, y_train, y_test = get_data('../p1_2/C_data_new.xls')
  # y_pred = model.predict([x1_test,x2_test])
  # # y_pred = np.round(y_pred,3)
  # y_pred = decode(y_pred)
  # me(y_test,y_pred)
  # results = np.concatenate([y_test,y_pred],axis=1)
  # results=np.round(results,3)
  # print(results.shape)
  #
  # data = pd.DataFrame(results,columns=[ f'{i}' for i in range(1,8)]+[ f'pred {i}' for i in range(1,8)])
  # data.to_csv('./pred_results.csv')
