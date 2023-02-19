# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_labels(data):
    # %%
    labels = data.loc[:,'1try':'7tries']
    labels = np.array(labels)
    # print(labels)
    return labels
def get_words_codes(data):
    # %%
    dict={}
    for i in range(26):
        dict[chr(97+i)]=i
    # print(dict)

    # %%
    words = data['Word']
    words_codes = []
    for word in words:
        codes=[]
        for letter in word:
            code = np.zeros((26))
            # print(letter)
            code[dict[letter]]=1
            codes.append(code)

        assert len(codes) ==5
        words_codes.append(codes)
        # print(words_codes)
    words_codes = np.array(words_codes)
    return words_codes
def get_freq(data):
    data_freq = data['word_rank']
    data_freq = np.array(data_freq)
    # mean = np.mean(data_freq)
    # var = np.var(data_freq)
    # data_freq = (data_freq-mean)/var*1e8
    data_freq =data_freq.reshape(-1,1)
    return  data_freq
def get_repeat(data):
    data_repeat = data['num of repeated letters']
    data_repeat = np.array(data_repeat)
    data_repeat = data_repeat
    data_repeat = data_repeat.reshape(-1,1)
    # print(data_repeat)
    return data_repeat
def get_letter_freq(data):

    data_letter_freq = data.loc[:,'first_entropy':'fifth_entropy']
    data_letter_freq = np.array(data_letter_freq).reshape(-1,5,1)
    return data_letter_freq
def get_all_entropy(data):

    data = data['all_entropy']
    all_entropy = np.array(data)
    all_entropy = np.array(all_entropy).reshape(-1,1)
    return all_entropy
def get_contest_number(data):
    data = data['Contest_number']
    contest_number = (np.array(data)-202)/420
    contest_number = np.array(contest_number).reshape(-1,1)
    return contest_number
def get_data(path):
    data = pd.read_excel(path)
    labels  = get_labels(data)




    words_codes = get_words_codes(data)#单词编码
    data_letter_freq = get_letter_freq(data)#字母信息熵序列

    data_freq = get_freq(data)#词频

    data_repeat = get_repeat(data)#单词字母重复次数

    all_entropy = get_all_entropy(data)#单词的信息熵
    contest_number = get_contest_number(data)



    x1 = words_codes
    x2 = np.concatenate([data_freq, data_repeat,all_entropy,contest_number], axis=1)
    x3 = data_letter_freq

    num = len(x1)
    # x1_train, x1_test, y_train, y_test = x1[:round(num*0.8)],x1[round(num*0.8):],labels[:round(num*0.8)],labels[round(num*0.8):]
    # x2_train,x2_test = x2[:round(num*0.8)],x2[round(num*0.8):]
    # x3_train,x3_test = x3[:round(num*0.8)],x3[round(num*0.8):]

    x1_train, x1_test, y_train, y_test = train_test_split(x1,labels,test_size=0.2,random_state=0)
    x2_train,x2_test, y_train, y_test  = train_test_split(x2,labels,test_size=0.2,random_state=0)
    x3_train,x3_test, y_train, y_test  = train_test_split(x3,labels,test_size=0.2,random_state=0)

    return x1_train, x1_test,x2_train,x2_test,x3_train,x3_test,y_train, y_test

if __name__ =='__main__':
    x1_train, x1_test,x2_train,x2_test,x3_train,x3_test,y_train, y_test= get_data('./c_data_new1 (2).xls')
    print(x1_train.shape,x2_train.shape,x1_test.shape)