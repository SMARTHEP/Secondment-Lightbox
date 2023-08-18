import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandas_ta

import tensorflow as tf
import yfinance as yf

from utils import prepare_dataframe, build_lstm_model


name = 'SPY'
train_end_date = '2022-01-01'

#download stock data the model has not seen (test set)
forecast_df = yf.download(name, start=train_end_date, end='2023-08-01')
forecast_df = forecast_df.reset_index().rename(columns={'index': 'Date'})

n_past = 14 #this must be the same as in the model training
n_future = 1 #this must be the same as in the model training
forecast_date_array = forecast_df['Date'].to_numpy()[n_past:]
truth_opens = forecast_df['Open'].values[n_past:]
truth_closes = forecast_df['Close'].values[n_past:]

columns_to_use = ['Open', 'High', 'Low', 'Close', 'Volume']
testX, testY, scaler = prepare_dataframe(forecast_df,columns_to_use,n_past,n_future)




#load the model and use it to predict the next Close
# model = build_lstm_model(input_shape=(n_past, len(columns_to_use)))
# Restore the weights
model_dir = "/Users/leonbozianu/work/lightbox/models/LSTM3DO_14past_1fut_15e_12515samples/"
model_save_loc = model_dir + "LSTM3DO_14past_1fut_15e_12515samples.keras"
# model.load_weights(model_save_loc)
model = tf.keras.models.load_model(model_save_loc)



forecast = model.predict(testX)
forecast_copies = np.repeat(forecast,len(columns_to_use),axis=-1)
y_pred_model = scaler.inverse_transform(forecast_copies)[:,3] #index 3 is Close

prediction_df = pd.DataFrame({'Date':forecast_date_array,'Close':truth_closes,'Close_pred':y_pred_model})


fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

ax[0].plot(prediction_df.Date,prediction_df.Close,label='Close',color='black')
ax[0].plot(prediction_df.Date,prediction_df.Close_pred,label=f'Close Pred TL',color='firebrick')
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[0].set_ylabel('price')
ax[0].set_title(name +' Stock (Test Set)')
ax[0].grid(color="0.95")

ax[1].plot(prediction_df.Date, prediction_df.Close_pred - prediction_df.Close, color='firebrick',label='$\Delta$ (pred - Close)',lw=0.75)
ax[1].axhline(0,ls='--',color='black')
ax[1].set_ylabel('Difference')
ax[1].grid(color="0.95")
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small')

fig.tight_layout()
fig.savefig(model_dir+"")


