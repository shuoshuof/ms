# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from datset import get_train_valid_data

# # %%
# data = data.iloc[1:,4]
# # print(data)
# data = np.array(data)
# # print(data)
#
# # %% [markdown]
# # + 使用60个过去的值预测预测60天后的值 比如1，60个数据 预测第120天的 因为题目的12月31号到3月1号刚好是60天
# # + 所以窗口大小为60
#
# # %%
# class Dataset:
#     def __init__(self,data,valid_rate=0.2,win_size=60,future=1,pred_len=10) -> None:
#         self.data = data
#         self.valid_rate = valid_rate
#         self.win_size = win_size
#         self.future = future
#         self.pred_len = pred_len
#         self.data_num =len(self.data)
#     def split_data(self,x,y):
#         x_train,x_valid = x[:round(self.data_num*self.valid_rate)],x[round(self.data_num*self.valid_rate):]
#         y_train,y_valid = y[:round(self.data_num*self.valid_rate)],y[round(self.data_num*self.valid_rate):]
#         return x_train,y_train,x_valid,y_valid
#     def normalize(self,data):
#         X=[]
#         Y=[]
#         for data_x,data_y in data:
#             x = [xx/data_x[0]-1 for xx in data_x]
#             y = [yy/data_x[0]-1 for yy in data_y]
#
#             X.append(x)
#             Y.append(y)
#         return np.array(X),np.array(Y)
#     def fnormalize(self,x,y,x0):
#         x = [(xx+1)*x0 for xx in x]
#         y = [(yy+1)*x0 for yy in y]
#         return x, y
#     def get_data(self):
#
#         x=[]
#         y=[]
#         for i in range(self.win_size,self.data_num-self.future-self.pred_len+2):
#             x.append(self.data[i-self.win_size:i])
#             y.append(self.data[i+self.future-1:i+self.future-1+self.pred_len])
#         x = np.array(x)
#         y = np.array(y)
#         x,y = self.normalize(zip(x,y))
#         return self.split_data(x,y)
#         # return x, y
#
#
# # %%
# lstm_data = Dataset(
#     data=data,
#
# )

# %%
x_train,y_train,x_valid,y_valid = get_train_valid_data('./Problem_C_Data_Wordle.xlsx')

print(x_train.shape, x_valid.shape)
# print(x_train)
#
# print(y_train)
# x,y = lstm_data.get_data()
# print(x,y)
# print(x.shape,y.shape)

# %%


# %%


class model:
    def __init__(self,input_shape,pred_len):
        self.input_shape = input_shape
        self.pred_len = pred_len
    def get_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(units=128,return_sequences=True,input_shape=self.input_shape))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.LSTM(units=128,return_sequences=True))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.LSTM(units=64,return_sequences=False))
        model.add(keras.layers.Dense(self.pred_len))
        model.compile(loss="mean_squared_error",optimizer="adam")
        model.summary()
        return model
input_shape = (60,1)
pred_len = 10
my_model = model(input_shape=input_shape,pred_len=pred_len).get_model()

save_path = './models_save/%s' % (time.strftime('%Y_%m_%d_%H_%M_%S'))

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=20,verbose=1)
early_stop =keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,verbose=1)
save_weights = keras.callbacks.ModelCheckpoint(save_path + "/model_{epoch:02d}_{val_loss:.4f}.h5",
                                                   save_best_only=True, monitor='val_loss')
callbacks_list = [reduce_lr,early_stop,save_weights]
my_model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_valid,y_valid),callbacks=callbacks_list)


