# -*- coding: utf-8 -*-
# @Time : 2023/2/18 12:40
# @Author : shuoshuo
# @File : predict.py
# @Project : ms
import numpy as np
from tensorflow import keras
from datset import *


def fnormalize(y, x0):
    y = [(yy + 1) * x0 for yy in y]
    return  y
def predict(model,data_x,contest_number,data):
    result_list = [[] for _ in range(561)]
    for i,(x,nums) in enumerate(zip(data_x,contest_number)):
        x0 = data[i]
        x = x.reshape(1,-1)

        y = model.predict(x)[0]

        y = fnormalize(y,x0)

        for yy,num in zip(y,nums):
            result_list[num].append(yy)
    draw_x = []
    results = []
    # print(result_list)
    for idx,ys in enumerate(result_list):
        # if len(ys)==10:
        results.append(np.sum(ys)/len(ys))
        draw_x.append(idx)
    return np.array(draw_x),np.array(results)
def predict_future(model,data,contest_number):
    future_results = [[] for _ in range(70)]
    t=0
    data= list(data)
    start_idx = 552-60-202
    reults = []
    print(len(data))
    while t<70:
        data_x = data[start_idx+t:start_idx+t+60]
        print(len(data_x))
        x = [xx / data_x[0]-1 for xx in data_x]
        t+=1
        x = np.array(x).reshape(1,-1)

        y = model.predict(x,verbose=0)[0]
        y = fnormalize(y, data_x[0])
        for i,yy in enumerate(y):
            if t+i-10>=0:
                future_results[t+i-10].append(yy)
        print(t,t%10)
        if t>=10:
            assert len(future_results[t-10])==10
            avg_y=np.sum(future_results[t-10])/10
            data.append(avg_y)
            reults.append(avg_y)
    return reults
def get_pre_data(path):
    data,contest_number = read_data(path)
    lstm_data = Dataset(
        data=data,
        contest_number=contest_number
    )
    x,y,num = lstm_data.get_data(split=False)
    return x,y,num
def draw(x,y,x_pred,y_pred):
    plt.figure()
    plt.plot(x,y)
    plt.plot(x_pred,y_pred)
    plt.show()
if __name__ == '__main__':
    model = keras.models.load_model('./models_save/2023_02_18_17_05_04/model_10_0.0096.h5')
    model.summary()

    x,y,num = get_pre_data('./Problem_C_Data_Wordle.xlsx')

    data, contest_number = read_data('./Problem_C_Data_Wordle.xlsx')
    draw_x,results=predict(model,x,num,data)
    #

    future_resluts = predict_future(model,data,contest_number)
    future_resluts=future_resluts[:60]

    draw_x = np.array(list(draw_x)+[draw_x[-1]+i for  i in range(60)])
    results= np.array(list(results)+future_resluts)
    print(len(results))
    draw(contest_number,data,draw_x,results)