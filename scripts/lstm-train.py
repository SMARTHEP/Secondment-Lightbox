import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import pandas_ta

import tensorflow as tf
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



#move to tensorflow, build stacked LSTM model
def build_lstm_model(input_shape):

    inputs = tf.keras.layers.Input(shape=input_shape)
    lstm_output1 = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    dropout_output1 = tf.keras.layers.Dropout(0.3)(lstm_output1)  # Adding dropout with a rate of 0.2
    lstm_output2 = tf.keras.layers.LSTM(128, return_sequences=True)(dropout_output1)
    dropout_output2 = tf.keras.layers.Dropout(0.3)(lstm_output2)  
    lstm_output3 = tf.keras.layers.LSTM(128, return_sequences=False)(dropout_output2)
    dropout_output3 = tf.keras.layers.Dropout(0.3)(lstm_output3)  

    flat_mean_output = tf.keras.layers.Dense(5, activation='relu')(dropout_output3) 
    output_mean = tf.keras.layers.Dense(1)(flat_mean_output)  # Output for mean prediction

    model = tf.keras.models.Model(inputs=inputs, outputs=output_mean)

    return model

# def custom_loss(y_true, y_pred):
#     # Calculate the MSE loss
#     mse_loss = tf.reduce_mean((tf.square(tf.subtract(y_true[1], y_pred))))
    
#     alpha = 0.01
#     # Calculate the trend consistency loss
#     true_trend = y_true[1] - y_true[0]
#     pred_trend = y_pred - y_true[0]
#     trend_loss = tf.reduce_mean(tf.square(tf.sign(true_trend) - tf.sign(pred_trend)))
    
#     # Combine the MSE loss and trend consistency loss with some weighting
#     combined_loss = mse_loss #+ alpha*trend_loss
    
#     return combined_loss

def custom_mean_squared_error(y_true, y_pred):
    squared_errors = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_errors)
    return mse



model = build_lstm_model(input_shape=(n_past, len(columns_to_use)))
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss=custom_mean_squared_error)
big_train_X = np.concatenate((trainX1,trainX2,trainX3,trainX4,trainX5,trainX),axis=0)
big_train_Y = np.concatenate((trainY1,trainY2,trainY3,trainY4,trainY5,trainY),axis=0)
# big_train_X = np.concatenate((trainX1,trainX2,trainX3,trainX4,trainX5,trainX6,trainX7,trainX8,trainX9,trainX10,trainX),axis=0)
# big_train_Y = np.concatenate((trainY1,trainY2,trainY3,trainY4,trainY5,trainY6,trainY7,trainY8,trainY9,trainY10,trainY),axis=0)


#complete training
history = model.fit(big_train_X, 
                    big_train_Y, 
                    epochs=12, 
                    batch_size=16, 
                    validation_split=0.3, 
                    verbose=1)

model_name = "LSTM3DO_{}past_{}fut_{}e_{}samples".format(n_past,n_future,history.params['epochs'],len(big_train_Y))
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
