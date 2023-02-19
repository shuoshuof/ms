# -*- coding: utf-8 -*-
# @Time : 2023/2/19 17:35
# @Author : shuoshuo
# @File : train.py
# @Project : ms
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
from dataset import *
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
def cal_cls_avg_try(data_tries,labels,cls_num):
    dict = {}
    for i in range(cls_num):
        idxs = np.where(labels==i)[0]
        i_tries_data = data_tries[idxs]
        avg_try = np.multiply(i_tries_data,np.arange(1,8))
        avg_try = np.sum(avg_try,axis=1)
        avg_avg_try = np.mean(avg_try)
        dict[i] = avg_avg_try
    return dict
def cal_cls_avg_repeat(data_repeat,labels,cls_num):
    dict = {}
    for i in range(cls_num):
        idxs = np.where(labels==i)[0]
        i_repeat_data = data_repeat[idxs]
        avg_repeat = np.mean(i_repeat_data)
        dict[i] = avg_repeat
    return dict
def cal_cls_avg_freq(data_word_freq,labels,cls_num):
    dict = {}
    for i in range(cls_num):
        idxs = np.where(labels==i)[0]
        i_freq_data = data_word_freq[idxs]
        avg_freq = np.mean(i_freq_data)
        dict[i] = avg_freq
    return dict
def cal_cls_num(labels,cls_num):
    dict = {}
    for i in range(cls_num):
        idxs = np.where(labels==i)[0]
        dict[i] = len(idxs)
    return dict
def plot_bar(dict_try,dict_repeat,dict_freq):
    # avg_try = dict_try.values()
    # avg_repeat = dict_repeat.values()
    # avg_freq = dict_freq.values()

    _,avg_try = zip(*dict_try)
    _,avg_repeat = zip(*dict_repeat)
    _,avg_freq = zip(*dict_freq)
    avg_try = np.array(avg_try)

    avg_repeat = np.array(avg_repeat)*10
    avg_freq = np.array(avg_freq) * 40
    print(avg_try)
    cls_names = ['Simple','Medium','Difficult','Hell']

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.3,0.8,0.6])

    width = 0.2
    ticks = np.arange(len(cls_names))

    ax.bar(ticks,avg_try,width,label='avg try')
    ax.bar(ticks+width,avg_repeat,width,label='avg repetition')
    ax.bar(ticks+2*width,avg_freq,width,label='avg frequency')
    ax.set_xticks(ticks+width/2)
    ax.set_xticklabels(cls_names)

    ax.legend(loc='best')
    plt.show()
def K_means(data_tries,data_tries_normal,data_word_freq,data_repeat):
    distortions = []
    sil_score = []
    k_nums= range(4,5)
    for cls_num in k_nums:

        model = KMeans(n_clusters=cls_num)

        # model.fit(data_tries_normal)
        # y_pred = model.predict(data_tries_normal)
        #
        # distortions.append(model.inertia_)
        # sil_score.append(silhouette_score(data_tries_normal,model.labels_))

        model.fit(data_tries)
        y_pred = model.predict(data_tries)

        distortions.append(model.inertia_)
        sil_score.append(silhouette_score(data_tries,model.labels_))


        dic_num = cal_cls_num(y_pred,cls_num)
        dict_try = cal_cls_avg_try(data_tries,y_pred,cls_num)
        dict_repeat = cal_cls_avg_repeat(data_repeat,y_pred,cls_num)
        dict_freq = cal_cls_avg_freq(data_word_freq,y_pred,cls_num)


        print('''''''')
        print(f'尝试次数{cls_num}')
        print("分布:",dic_num)
        dict_try = sorted(dict_try.items(),key=lambda x:x[1])#从易到难，平均尝试次数越少越简单
        print("平均尝试次数:",dict_try)
        dict_repeat = sorted(dict_repeat.items(),key=lambda x:x[1])
        print("平均重复次数:",dict_repeat)
        dict_freq = sorted(dict_freq.items(),key=lambda x:x[1],reverse=True)
        print("平均词频:",dict_freq)
        print('''''''')
        # avg_try = zip(*zip(dict_try))
        # avg_repeat = dict(dict_repeat
        # avg_freq = dict_freq.values()
        plot_bar(dict_try, dict_repeat, dict_freq)
    plt.plot(k_nums, distortions, marker='o', )
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

    plt.figure()
    plt.plot(k_nums, sil_score, marker='o', )
    plt.xlabel('Number of clusters')
    plt.ylabel('silhouette score')
    plt.show()
if __name__ == '__main__':
    data_tries,data_tries_normal,data_word_freq,data_repeat = get_data()
    K_means(data_tries,data_tries_normal,data_word_freq,data_repeat)

    # dict_try = cal_cls_avg_try(data_tries,y_pred,3)
    # dict_repeat = cal_cls_avg_repeat(data_repeat,y_pred,3)
    # dict_freq = cal_cls_avg_freq(data_word_freq,y_pred,3)
    # print(dict_try)
    # print(dict_repeat)
    # print(dict_freq)