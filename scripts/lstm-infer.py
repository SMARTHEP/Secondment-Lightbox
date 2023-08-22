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
model = build_lstm_model(input_shape=(n_past, len(columns_to_use)))
# Restore the weights
model_dir = "/Users/leonbozianu/work/lightbox/models/LSTM3DO_14past_1fut_12e_15018samples"
model_save_loc = model_dir + "/final_weight.h5"
model.load_weights(model_save_loc)

forecast = model.predict(testX)
y_pred_model = scaler['Close'].inverse_transform(forecast).squeeze()

prediction_df = pd.DataFrame({'Date':forecast_date_array,'Close':truth_closes,'Close_pred':y_pred_model})

print('Saving plots in folder here:',model_dir)
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

ax[0].plot(prediction_df.Date,prediction_df.Close,label='Close',color='black')
# ax[0].plot(prediction_df.Date,prediction_df.Close.shift(1),label='No model',color='grey')
ax[0].plot(prediction_df.Date,prediction_df.Close_pred,label=f'Model Pred',color='firebrick')
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[0].set_ylabel('price')
ax[0].set_title(name +' Stock (Test Set)')
ax[0].grid(color="0.95")

# ax[1].plot(prediction_df.Date, prediction_df.Close.shift(1) - prediction_df.Close, color='grey',label='No model',lw=0.75)
ax[1].plot(prediction_df.Date, prediction_df.Close_pred - prediction_df.Close, color='firebrick',label='Model Pred',lw=0.75)
ax[1].axhline(0,ls='--',color='black')
ax[1].set_ylabel('$\Delta$ (pred - Close)')
ax[1].grid(color="0.95")
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small')

fig.tight_layout()
fig.savefig(model_dir+"/test_set_Close.png")



prediction_df['pct_change'] = prediction_df['Close'].pct_change().fillna(0)
prediction_df['pct_change_pred'] = (prediction_df['Close_pred'] - prediction_df['Close']) / prediction_df['Close']

plt.figure()
plt.scatter(prediction_df['pct_change'], prediction_df['pct_change_pred'],color='orange',alpha=.4)
plt.plot(prediction_df['pct_change'], prediction_df['pct_change'],lw=4,color='cadetblue')
plt.axhline(y=0,ls='--',color='red')
plt.axvline(x=0,ls='--',color='red')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.savefig(model_dir+"/test_set_scatter.png")