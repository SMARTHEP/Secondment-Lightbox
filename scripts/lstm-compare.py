import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandas_ta

import tensorflow as tf
import yfinance as yf

from utils import prepare_dataframe, build_lstm_model

name = 'MSFT'
train_end_date = '2022-01-01'

#download stock data the model has not seen (test set)
forecast_df = yf.download(name, start=train_end_date, end='2023-08-01')
forecast_df = forecast_df.reset_index().rename(columns={'index': 'Date'})

#prepare data, n_past can change here, but we need the largest n_past as index for truth_closes
n_past = 14 
n_future = 1 #this must be the same as in the model training
forecast_date_array = forecast_df['Date'].to_numpy()[n_past:]
truth_closes = forecast_df['Close'].values[n_past:]

columns_to_use = ['Open', 'High', 'Low', 'Close', 'Volume']
testX, testY, scaler = prepare_dataframe(forecast_df,columns_to_use,n_past,n_future)

#now to retrieve the models
model_locations = ['/Users/leonbozianu/work/lightbox/models/LSTM3DO_14past_1fut_15e_12515samples',
                   '/Users/leonbozianu/work/lightbox/models/LSTM3DO_14past_1fut_8e_15018samples',
                   '/Users/leonbozianu/work/lightbox/models/LSTM3DO_14past_1fut_15e_15018samples',
                   '/Users/leonbozianu/work/lightbox/models/LSTM3DO_14past_3fut_15e_5004samples',
                   '/Users/leonbozianu/work/lightbox/models/LSTM3DO_14past_6fut_15e_2502samples',
                   '/Users/leonbozianu/work/lightbox/models/LSTM3DO_20past_1fut_15e_14982samples']


#plotting
save_loc = "/Users/leonbozianu/work/lightbox/compare"
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
ax[0].plot(forecast_date_array,truth_closes,label='Close',color='black')

labels = ['12k Train', '8e 15k Train', '15k Train','Step 3','Step 6','Lookback 20']
for i in range(len(model_locations)):
    model = build_lstm_model(input_shape=(n_past, len(columns_to_use)))
    model.load_weights(model_locations[i]+ "/final_weight.h5")  
    forecast = model.predict(testX)
    y_pred_model = scaler['Close'].inverse_transform(forecast).squeeze()

    ax[0].plot(forecast_date_array,y_pred_model,label=labels[i])    
    ax[1].plot(forecast_date_array, y_pred_model - truth_closes,label=labels[i],lw=0.75)


ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small')
ax[0].set_ylabel('price')
ax[0].set_title(name +' Stock (Test Set)')
ax[0].grid(color="0.95")

ax[1].axhline(0,ls='--',color='black')
ax[1].set_ylabel('$\Delta$ (pred - Close)')
ax[1].grid(color="0.95")
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small')

fig.tight_layout()
fig.savefig(save_loc+"/model_compare.png")














