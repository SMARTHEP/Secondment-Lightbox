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

#prepare data, n_past can change here, but we need the largest n_past as index for truth_closes
n_past = 14 
n_future = 1 #this must be the same as in the model training
forecast_date_array = forecast_df['Date'].to_numpy()[n_past:]
truth_closes = forecast_df['Close'].values[n_past:]

columns_to_use = ['Open', 'High', 'Low', 'Close', 'Volume']
testX, testY, scaler = prepare_dataframe(forecast_df,columns_to_use,n_past,n_future)


#now to retrieve the models
model_locations = ['/Users/leonbozianu/work/lightbox/models/LSTM3DO_14past_1fut_15e_12515samples/',
                   '/Users/leonbozianu/work/lightbox/models/LSTM3DO_14past_1fut_15e_12515samples/']



















