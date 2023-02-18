# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_labels(data):
    # %%
    labels = data.iloc[:,7:14]
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
    data = pd.read_excel('../p1_2/C_data (2).xls')
    data_freq = data['Count']
    data_freq = np.array(data_freq)
    mean = np.mean(data_freq)
    var = np.var(data_freq)
    data_freq = (data_freq-mean)/var*1e8
    data_freq =data_freq.reshape(-1,1)
    return  data_freq
def get_repeat(data):
    data_repeat = data['num of repeated letters']
    data_repeat = np.array(data_repeat)
    data_repeat = data_repeat-1
    data_repeat = data_repeat.reshape(-1,1)
    # print(data_repeat)
    return data_repeat
def get_data(path):
    data = pd.read_excel(path)
    labels  = get_labels(data)

    words_codes = get_words_codes(data)
    data_freq = get_freq(data)
    data_repeat = get_repeat(data)

    x1 = words_codes
    x2 = np.concatenate([data_freq, data_repeat], axis=1)
    num = len(x1)
    x1_train, x1_test, y_train, y_test = x1[:round(num*0.8)],x1[round(num*0.8):],labels[:round(num*0.8)],labels[round(num*0.8):]
    x2_train,x2_test = x2[:round(num*0.8)],x2[round(num*0.8):]
    return x1_train, x1_test,x2_train,x2_test, y_train, y_test

if __name__ =='__main__':
    x1_train, x1_test,x2_train,x2_test, y_train, y_test= get_data('../p1_2/C_data_new.xls')
    print(x1_train.shape,x2_train.shape,x1_test.shape)