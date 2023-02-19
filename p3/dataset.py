# -*- coding: utf-8 -*-
# @Time : 2023/2/19 13:07
# @Author : shuoshuo
# @File : dataset.py
# @Project : ms

import numpy as np
import pandas as pd

def try_normalize(data):
    var = np.var(data,axis=0)
    mean = np.mean(data,axis=0)
    data= (data-mean)/var

    # min = np.min(data,axis=0)
    # max = np.max(data,axis=0)
    # data = (data-min)/(max-min)
    return data

def get_data(path='../p2/C_data_new.xls'):
    data = pd.read_excel(path)
    data_tries = data.iloc[:,7:14]
    data_tries = np.array(data_tries)
    data_tries_normal = try_normalize(data_tries)

    data_word_freq = data['word_freq']
    data_word_freq = np.array(data_word_freq).reshape(-1,1)

    data_repeat = data['num of repeated letters']
    data_repeat = np.array(data_repeat).reshape(-1,1)
    return data_tries,data_tries_normal,data_word_freq,data_repeat

if __name__ == '__main__':
    data_tries,data_tries_normal,data_word_freq,data_repeat = get_data()
