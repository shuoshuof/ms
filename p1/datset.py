# -*- coding: utf-8 -*-
# @Time : 2023/2/18 12:45
# @Author : shuoshuo
# @File : datset.py
# @Project : ms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, data,contest_number, valid_rate=0.2, win_size=60, future=1, pred_len=10) -> None:
        self.data = data
        self.contest_number = contest_number
        self.valid_rate = valid_rate
        self.win_size = win_size
        self.future = future
        self.pred_len = pred_len
        self.data_num = len(self.data)

    def split_data(self, x, y):
        data_len = len(x)
        x_train, x_valid = x[:round(data_len * (1-self.valid_rate))], x[round(data_len * (1-self.valid_rate)):]
        y_train, y_valid = y[:round(data_len *(1-self.valid_rate))], y[round(data_len * (1-self.valid_rate)):]
        return x_train, y_train, x_valid, y_valid

    def normalize(self, data):
        X = []
        Y = []
        for data_x, data_y in data:
            x = [xx / data_x[0] - 1 for xx in data_x]
            y = [yy / data_x[0] - 1 for yy in data_y]

            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)

    def fnormalize(self, x, y, x0):
        x = [(xx + 1) * x0 for xx in x]
        y = [(yy + 1) * x0 for yy in y]
        return x, y

    def get_data(self):

        x = []
        y = []
        for i in range(self.win_size, self.data_num - self.future - self.pred_len + 2):
            x.append(self.data[i - self.win_size:i])
            y.append(self.data[i + self.future - 1:i + self.future - 1 + self.pred_len])
        x = np.array(x)
        y = np.array(y)
        x, y = self.normalize(zip(x, y))
        return self.split_data(x, y)
def read_data(path):
    data = pd.read_excel('./Problem_C_Data_Wordle.xlsx')
    data = data.iloc[1:, 4]
    data = np.array(data)
    data = data[::-1]
    return data

def get_train_valid_data(path):
    data = read_data(path)

    lstm_data = Dataset(
        data=data,
    )
    x_train, y_train, x_valid, y_valid = lstm_data.get_data()
    return x_train, y_train, x_valid, y_valid
def draw(x,y):
    plt.figure()
    plt.plot(x,y)
    plt.show()
if __name__ == '__main__':

    x_train, y_train, x_valid, y_valid = get_train_valid_data('./Problem_C_Data_Wordle.xlsx')
    print(x_train.shape,x_valid.shape)
    data = read_data('./Problem_C_Data_Wordle.xlsx')
    x= np.arange(len(data))
    draw(x,data)