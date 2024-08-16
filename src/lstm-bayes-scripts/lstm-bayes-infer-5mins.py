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





# CL = pd.read_excel("/Users/leonbozianu/work/lightbox-legacy/CL.xlsx")
CL = pd.read_excel("/Users/leonbozianu/work/lightbox-legacy/CL.xlsx",sheet_name='5_mins')
# df_sheet_multi = pd.read_excel('CL.xlsx', sheet_name=[0, 'sheet2'])

name_ohlc_df = CL.reset_index()
data = {'date': name_ohlc_df.date, 'close': name_ohlc_df.close, 'high': name_ohlc_df.high, 'low': name_ohlc_df.low, 'volume': name_ohlc_df.volume}
df = pd.DataFrame(data)
print(df.shape)
print(df.head())
print(df.tail())

# concatenated_df = pd.concat([df1, df2], axis=0, ignore_index=True)
# train_size = int(0.9 * len(df))
train_size = len(df) - 250
print('TRAIN UNTIL:',df.iloc[train_size,0],'\tTEST UNTIL:',df.iloc[-1,0],'\n')

sequence_length = 10
train_input = np.lib.stride_tricks.sliding_window_view(df.iloc[:train_size,1:-1].values, (sequence_length,3)).squeeze()
val_input = np.lib.stride_tricks.sliding_window_view(df.iloc[train_size:-1,1:-1].values, (sequence_length,3)).squeeze()

y_train = np.array(df.iloc[sequence_length:train_size+1,1]).T.astype(np.float32).reshape(-1,1)
y_val = np.array(df.iloc[sequence_length+train_size:,1]).T.astype(np.float32).reshape(-1,1)

deep_model = tf.keras.Sequential([
  tf.keras.layers.LSTM(32,activation='selu',input_shape=(sequence_length,3),return_sequences=True),
  tf.keras.layers.LSTM(64,activation='selu',return_sequences=False),
  tf.keras.layers.Dense(128,activation='selu'),
  tf.keras.layers.Dense(64,activation='selu'),
  tf.keras.layers.Dense(24,activation='selu'),
  tf.keras.layers.Dense(8,activation='selu'),
  tf.keras.layers.Dense(1+1),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],scale=1e-3 + tf.math.softplus(0.001*t[...,1:]))),
      # lambda t: tfd.StudentT(df=4,loc=t[..., :1],scale=1e-3 + tf.math.softplus(t[...,1:])))
])


negloglik = lambda y, rv_y: -rv_y.log_prob(y)
deep_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss=negloglik)
# Restore the weights
model_name = "model-5-mins"
model_save_loc = "/Users/leonbozianu/work/lightbox/models/{}".format(model_name) + "/weights-5-mins.h5"
deep_model.load_weights(model_save_loc)


yhat = deep_model(val_input)
assert isinstance(yhat, tfd.Distribution)


yhats = [deep_model(val_input) for _ in range(100)]
for i, yhat in enumerate(yhats):
  m = np.squeeze(yhat.mean())
  s = np.squeeze(yhat.stddev())


val_dates = np.array(df.iloc[sequence_length+train_size:,0]).T.reshape(-1,1)
print(val_dates)
print(val_dates.shape,y_val.shape,m.shape)
upper_bound = m + s
lower_bound = m - s

plt.figure(figsize=(10, 6))
plt.plot(val_dates,y_val, label='True Values', color='black',marker='o',ms=2.5)
plt.plot(val_dates,m, label='Predicted Mean', color='red',marker='x',ms=2.5)
plt.fill_between(val_dates.reshape(-1), lower_bound, upper_bound, color='orange', alpha=0.3, label='Uncertainty Interval')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title(f'5 Minute Predictions with Uncertainties (Trained to {df.date.tolist()[train_size]})')
plt.legend()
plt.grid(True)
plt.tight_layout()
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


mape = calculate_mape(y_val, m)
rmse = calculate_rmse(y_val, m)
mae = calculate_mae(y_val, m)
fb = calculate_forecast_bias(y_val, m)
r2 = calculate_r2(y_val, m)
print("MAPE:", mape,'\nRMSE:',rmse,'\nMAE:',mae,'\nFB:',fb,'\nR2:',r2)
print()

y_val = y_val.reshape(-1)
true_pct_changes = (np.diff(y_val) / y_val[:-1]) * 100
pred_pct_changes = ((m - y_val) / y_val) * 100 

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(np.sign(true_pct_changes), np.sign(pred_pct_changes[:-1]), labels=[1, -1])
print("Confusion Matrix:")
print(conf_matrix)





#strategy
#find the days were the price today is below -2sigma or above +2sigma
print(val_dates.shape,y_val.shape,m.shape,s.shape)
prediction_df = pd.DataFrame({'Date':val_dates.squeeze(),'Close':y_val.squeeze(), 
                              'Model_mean':m, 'Model_sigma':s})
level = 1
prediction_df[f'm+{level}s'] = prediction_df['Model_mean'] + level*prediction_df['Model_sigma']
prediction_df[f'm-{level}s'] = prediction_df['Model_mean'] - level*prediction_df['Model_sigma']
condition1 = prediction_df['Close'] > prediction_df[f'm+{level}s']
condition2 = prediction_df['Close'] < prediction_df[f'm-{level}s']

above_result = prediction_df[condition1]
above_result_dates = above_result['Date'].tolist()
below_result = prediction_df[condition2]
below_result_dates = below_result['Date'].tolist()



plt.figure(figsize=(10, 6))
plt.plot(prediction_df['Date'],prediction_df['Close'], label='True Values', color='black',marker='o',ms=3)
plt.plot(prediction_df['Date'],prediction_df['Model_mean'], label='Predicted Mean', color='red',marker='x',ms=3)
plt.fill_between(prediction_df['Date'], prediction_df[f'm-{level}s'], prediction_df[f'm+{level}s'], color='orange', alpha=0.3, label=f'{level}$\sigma$ Uncertainty')
if len(above_result_dates)>0:
    plt.axvline(above_result_dates[0], color='g', linestyle='--', label='Above Cond. Met',alpha=0.3)
    # plt.vlines(above_result_dates,ymin=min(prediction_df['Close']), ymax=max(prediction_df['Close']), color='g', linestyle='--', label='Above Cond. Met',alpha=0.3)
if len(below_result_dates)>0:
    plt.axvline(below_result_dates[0], color='r', linestyle='--', label='Below Cond. Met',alpha=0.3)
    # plt.vlines(below_result_dates,ymin=min(prediction_df['Close']), ymax=max(prediction_df['Close']), color='r', linestyle='--', label='Below Cond. Met',alpha=0.3)

combined_x_values = above_result_dates + below_result_dates
if len(combined_x_values)>0:
    color_indicator = [1] * len(above_result_dates) + [-1] * len(below_result_dates)
    combined_x_values, color_indicator = zip(*sorted(zip(combined_x_values, color_indicator)))
    current_color = None  # Initialize with None
    for x, color_code in zip(combined_x_values, color_indicator):
        color = 'green' if color_code == 1 else 'red'
        plt.vlines(x, ymin=min(prediction_df['Close']), ymax=max(prediction_df['Close']), color=color,ls='--',alpha=0.3)
        if current_color is not None:
            plt.fill_betweenx([min(prediction_df[f'm-{level}s']), max(prediction_df[f'm+{level}s'])], previous_x, x, color=current_color, alpha=0.2) 
        current_color = color
        previous_x = x
    plt.fill_betweenx([min(prediction_df[f'm-{level}s']), max(prediction_df[f'm+{level}s'])], previous_x, max(prediction_df['Date']), color=current_color, alpha=0.2) 
        
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title(f'5 Minute Predictions with Uncertainties (Trained to {df.date.tolist()[train_size]})')
plt.legend()
plt.grid(color="0.5")
plt.tight_layout()
plt.show()