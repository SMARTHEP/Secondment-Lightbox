import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import yfinance as yf
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions



name = 'AAPL'

data_end_date = '2014-08-01'
name_ohlc_df = yf.download(name, start='2002-01-01', end=data_end_date)
name_ohlc_df = name_ohlc_df.reset_index()
data = {'date': name_ohlc_df.Date, 'close': name_ohlc_df.Close, 'volume': name_ohlc_df.Volume}
df = pd.DataFrame(data)

# concatenated_df = pd.concat([df1, df2], axis=0, ignore_index=True)
train_size = int(0.9 * len(df))
print('TRAIN UNTIL:',df.iloc[train_size],'\tTEST UNTIL:',df.iloc[-1],'\n\n\n')

sequence_length = 10
train_input = np.lib.stride_tricks.sliding_window_view(df.iloc[:train_size,1].values, (sequence_length,))
val_input = np.lib.stride_tricks.sliding_window_view(df.iloc[train_size:-1,1].values, (sequence_length,))

y1 = np.array(df.iloc[sequence_length:train_size+1,1]).T.astype(np.float32).reshape(-1,1)
y2 = np.array(df.iloc[sequence_length+train_size:,1]).T.astype(np.float32).reshape(-1,1)

bayes_lstm_model = tf.keras.Sequential([
  tf.keras.layers.LSTM(32,activation='relu',input_shape=(sequence_length,1),return_sequences=False),
  tf.keras.layers.Dense(64,activation='relu'),
  tf.keras.layers.Dense(24,activation='relu'),
  tf.keras.layers.Dense(8,activation='relu'),
  tf.keras.layers.Dense(1+1),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],scale=1e-3 + tf.math.softplus(0.001*t[...,1:]))),
    #   lambda t: tfd.StudentT(df=4,loc=t[..., :1],scale=1e-3 + tf.math.softplus(t[...,1:])))
])




negloglik = lambda y, rv_y: -rv_y.log_prob(y)
bayes_lstm_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss=negloglik)
# Restore the weights
model_name = "model-{}".format(data_end_date)
model_save_loc = "/Users/leonbozianu/work/lightbox/models/{}".format(model_name) + "/weights-end-date-{}.h5".format(data_end_date)
bayes_lstm_model.load_weights(model_save_loc)


yhat = bayes_lstm_model(val_input)
assert isinstance(yhat, tfd.Distribution)


yhats = [bayes_lstm_model(val_input) for _ in range(100)]
for i, yhat in enumerate(yhats):
  m = np.squeeze(yhat.mean())
  s = np.squeeze(yhat.stddev())


val_dates = np.array(df.iloc[sequence_length+train_size:,0]).T.reshape(-1,1)
print(val_dates.shape,y2.shape,m.shape)
upper_bound = m + s
lower_bound = m - s

plt.figure(figsize=(10, 6))
plt.plot(val_dates,y2, label='True Values', color='black')
plt.plot(val_dates,m, label='Predicted Mean', color='red')
plt.fill_between(val_dates.reshape(-1), lower_bound, upper_bound, color='orange', alpha=0.3, label='Uncertainty Interval')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title(f'{name} Predictions with Uncertainties (Trained to {df.date.tolist()[train_size]})')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.xlim((val_dates[50][0],val_dates[350][0]))
plt.show()






from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_mape(true_data, forecast_data):
    absolute_percentage_errors = np.abs((true_data - forecast_data) / true_data)
    mape = np.mean(absolute_percentage_errors) * 100
    return mape

def calculate_rmse(true_data, forecast_data):
    rmse = np.sqrt(mean_squared_error(true_data, forecast_data))
    return rmse

def calculate_mae(true_data, forecast_data):
    mae = mean_absolute_error(true_data, forecast_data)
    return mae

def calculate_r2(true_data, forecast_data):
    r2 = r2_score(true_data, forecast_data)
    return r2

def calculate_forecast_bias(true_data, forecast_data):
    forecast_bias = np.mean(forecast_data - true_data)
    return forecast_bias


mape = calculate_mape(y2, m)
rmse = calculate_rmse(y2, m)
mae = calculate_mae(y2, m)
fb = calculate_forecast_bias(y2, m)
r2 = calculate_r2(y2, m)
print("MAPE:", mape,'\nRMSE:',rmse,'\nMAE:',mae,'\nFB:',fb,'\nR2:',r2)
print()


y2 = y2.reshape(-1)
true_pct_changes = (np.diff(y2) / y2[:-1]) * 100
pred_pct_changes = ((m - y2) / y2) * 100 

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(np.sign(true_pct_changes), np.sign(pred_pct_changes[:-1]), labels=[1, -1])
print("Confusion Matrix:")
print(conf_matrix)

#strategy
#find the days were the price today is below -2sigma or above +2sigma
print(val_dates.shape,y2.shape,m.shape,s.shape)
prediction_df = pd.DataFrame({'Date':val_dates.squeeze(),'Close':y2.squeeze(), 
                              'Model_mean':m, 'Model_sigma':s})
level = 3
prediction_df[f'm+{level}s'] = prediction_df['Model_mean'] + level*prediction_df['Model_sigma']
prediction_df[f'm-{level}s'] = prediction_df['Model_mean'] - level*prediction_df['Model_sigma']
prediction_df.head()


condition1 = prediction_df['Close'] > prediction_df[f'm+{level}s']
condition2 = prediction_df['Close'] < prediction_df[f'm-{level}s']

above_result = prediction_df[condition1]
above_result_dates = above_result['Date'].tolist()
below_result = prediction_df[condition2]
below_result_dates = below_result['Date'].tolist()




plt.figure(figsize=(10, 6))
plt.plot(prediction_df['Date'],prediction_df['Close'], label='True Values', color='black')
plt.plot(prediction_df['Date'],prediction_df['Model_mean'], label='Predicted Mean', color='red')
plt.fill_between(prediction_df['Date'], prediction_df[f'm-{level}s'], prediction_df[f'm+{level}s'], color='indigo', alpha=0.3, label=f'{level}$\sigma$ Uncertainty')
plt.axvline(above_result_dates[0], color='g', linestyle='--', label='Above Cond. Met',alpha=0.3)
plt.axvline(below_result_dates[0], color='r', linestyle='--', label='Below Cond. Met',alpha=0.3)

combined_x_values = above_result_dates + below_result_dates
color_indicator = [1] * len(above_result_dates) + [-1] * len(below_result_dates)
combined_x_values, color_indicator = zip(*sorted(zip(combined_x_values, color_indicator)))
current_color = None  # Initialize with None
for x, color_code in zip(combined_x_values, color_indicator):
    color = 'green' if color_code == 1 else 'red'
    plt.vlines(x, ymin=min(prediction_df['Close']), ymax=max(prediction_df['Close']), color=color, lw=0.5,ls='--',alpha=0.3)
    if current_color is not None:
        plt.fill_betweenx([min(prediction_df['Close']), max(prediction_df['Close'])], previous_x, x, color=current_color, alpha=0.2) 
    current_color = color
    previous_x = x
plt.fill_betweenx([min(prediction_df['Close']), max(prediction_df['Close'])], previous_x, max(prediction_df['Date']), color=current_color, alpha=0.2) 
    
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title(f'{name} Predictions with Uncertainties (Trained to {df.date.tolist()[train_size]})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





