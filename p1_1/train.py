# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from datset import get_train_valid_data



x,y,num = get_train_valid_data('./Problem_C_Data_Wordle.xlsx',split=False)

print(x.shape, y.shape)

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
# my_model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_valid,y_valid),callbacks=callbacks_list)
my_model.fit(x,y,epochs=10,batch_size=32,callbacks=[keras.callbacks.ModelCheckpoint(save_path + "/model_{epoch:02d}_{loss:.4f}.h5",
                                                   save_best_only=False,monitor='loss')])

