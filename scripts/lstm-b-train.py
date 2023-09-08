import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import pandas_ta

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import yfinance as yf

from utils import prepare_dataframe

#download stock data using yahoo finance
name = 'SPY'
train_end_date = '2022-01-01'

name_ohlc_df = yf.download(name, start='2012-01-01', end=train_end_date)
spy_ohlc_df = yf.download('SPY', start='2012-01-01', end=train_end_date)
msft_ohlc_df = yf.download('MSFT', start='2012-01-01', end=train_end_date)
aapl_ohlc_df = yf.download('AAPL', start='2012-01-01', end=train_end_date)
goog_ohlc_df = yf.download('GOOG', start='2012-01-01', end=train_end_date)
meta_ohlc_df = yf.download('META', start='2012-01-01', end=train_end_date)
nvda_ohlc_df = yf.download('NVDA', start='2012-01-01', end=train_end_date)
intc_ohlc_df = yf.download('INTC', start='2012-01-01', end=train_end_date)
amd_ohlc_df = yf.download('AMD', start='2012-01-01', end=train_end_date)
amzn_ohlc_df = yf.download('AMZN', start='2012-01-01', end=train_end_date)
baba_ohlc_df = yf.download('BABA', start='2012-01-01', end=train_end_date)

#prepare dataframes, then convert to appropriate numpy
df = name_ohlc_df.copy()
df = df.reset_index().rename(columns={'index': 'Date'})
date_array = df['Date'].to_numpy()
date_series = df['Date']

cols = list(df)[1:]
print(cols) #['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
df_for_training = df[cols].astype(float)
df_for_training1 = spy_ohlc_df[cols].astype(float)
df_for_training2 = msft_ohlc_df[cols].astype(float)
df_for_training3 = aapl_ohlc_df[cols].astype(float)
df_for_training4 = goog_ohlc_df[cols].astype(float)
df_for_training5 = amzn_ohlc_df[cols].astype(float)
df_for_training6 = meta_ohlc_df[cols].astype(float)
df_for_training7 = nvda_ohlc_df[cols].astype(float)
df_for_training8 = intc_ohlc_df[cols].astype(float)
df_for_training9 = intc_ohlc_df[cols].astype(float)
df_for_training10 = baba_ohlc_df[cols].astype(float)

n_past = 14
n_future = 1
columns_to_use = ['Open', 'High', 'Low', 'Close', 'Volume']
trainX, trainY, _ = prepare_dataframe(df_for_training,columns_to_use,n_past,n_future)
trainX1, trainY1, _ = prepare_dataframe(df_for_training1,columns_to_use,n_past,n_future)
trainX2, trainY2, _ = prepare_dataframe(df_for_training2,columns_to_use,n_past,n_future)
trainX3, trainY3, _ = prepare_dataframe(df_for_training3,columns_to_use,n_past,n_future)
trainX4, trainY4, _ = prepare_dataframe(df_for_training4,columns_to_use,n_past,n_future)
trainX5, trainY5, _ = prepare_dataframe(df_for_training5,columns_to_use,n_past,n_future)
trainX6, trainY6, _ = prepare_dataframe(df_for_training6,columns_to_use,n_past,n_future)
trainX7, trainY7, _ = prepare_dataframe(df_for_training7,columns_to_use,n_past,n_future)
trainX8, trainY8, _ = prepare_dataframe(df_for_training8,columns_to_use,n_past,n_future)
trainX9, trainY9, _ = prepare_dataframe(df_for_training9,columns_to_use,n_past,n_future)
trainX10, trainY10, _ = prepare_dataframe(df_for_training10,columns_to_use,n_past,n_future)







#move to tensorflow, build bayesian LSTM model
def build_bayes_model(input_shape):
    
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=(n_past, len(columns_to_use))))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.LSTM(units=100, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.LSTM(units=100, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.3))

    # model.add(tf.keras.layers.Dense(units=5))
    # model.add(tf.keras.layers.Dense(units=1))
    model.add(tf.keras.layers.Dense(20,activation="relu"))
    model.add(tf.keras.layers.Dense(2))
    model.add(tfp.layers.DistributionLambda(lambda t:
                tfd.Normal(loc   = t[...,:1],
                           scale=tf.math.softplus(0.005*t[...,1:])+0.001)))

    return model

negloglik = lambda y, p_y: -p_y.log_prob(y)


model = build_bayes_model(input_shape=(n_past, len(columns_to_use)))
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.1), loss=negloglik)
big_train_X = np.concatenate((trainX1,trainX2,trainX3,trainX4,trainX5,trainX),axis=0)
big_train_Y = np.concatenate((trainY1,trainY2,trainY3,trainY4,trainY5,trainY),axis=0)
# big_train_X = np.concatenate((trainX1,trainX2,trainX3,trainX4,trainX5,trainX6,trainX7,trainX8,trainX9,trainX10,trainX),axis=0)
# big_train_Y = np.concatenate((trainY1,trainY2,trainY3,trainY4,trainY5,trainY6,trainY7,trainY8,trainY9,trainY10,trainY),axis=0)


#complete training
history = model.fit(big_train_X, 
                    big_train_Y, 
                    epochs=50, 
                    batch_size=16, 
                    validation_split=0.2, 
                    verbose=1)

model_name = "LSTM_B_3DO_{}past_{}fut_{}e_{}samples".format(n_past,n_future,history.params['epochs'],len(big_train_Y))
model_folder = '/Users/leonbozianu/work/lightbox/models/{}'.format(model_name)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model.save_weights(filepath=model_folder+'/final_weight.h5')



#plot training and validation losses over 
plt.figure()
plt.plot(history.history['loss'],'--', label='Train loss TL')
plt.plot(history.history['val_loss'],'--', label='Val lossTL')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small')
plt.xlabel('Num. Epochs')
plt.ylabel('MSE Loss')
plt.grid()
plt.tight_layout()
plt.savefig('/Users/leonbozianu/work/lightbox/models/{}/train_val_loss.png'.format(model_name))
