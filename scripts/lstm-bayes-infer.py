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

train_end_date = '2023-07-01'
name_ohlc_df = yf.download(name, start='2009-01-01', end=train_end_date)
name_ohlc_df = name_ohlc_df.reset_index()

data = {'date': name_ohlc_df.Date, 'close': name_ohlc_df.Close, 'volume': name_ohlc_df.Volume}
df = pd.DataFrame(data)


train_size = int(0.75 * len(df))


sequence_length = 10
train_input = np.lib.stride_tricks.sliding_window_view(df.iloc[:train_size,1].values, sequence_length)
val_input = np.lib.stride_tricks.sliding_window_view(df.iloc[train_size:-1,1].values, sequence_length)

y1 = np.array(df.iloc[sequence_length:train_size+1,1]).T.astype(np.float32).reshape(-1,1)
y2 = np.array(df.iloc[sequence_length+train_size:,1]).T.astype(np.float32).reshape(-1,1)




dmodel = tf.keras.Sequential([
  tf.keras.layers.LSTM(32,activation='relu',input_shape=(10,1)),
  tf.keras.layers.Dense(64,activation='relu'),
  tf.keras.layers.Dense(24,activation='relu'),
  tf.keras.layers.Dense(8,activation='relu'),
  tf.keras.layers.Dense(1+1),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1],scale=1e-3 + tf.math.softplus(0.1 * t[...,1:]))),
])

negloglik = lambda y, rv_y: -rv_y.log_prob(y)
dmodel.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss=negloglik)
# Restore the weights
model_save_loc = "/Users/leonbozianu/work/lightbox/models/model_bayes" + "/final_weight.h5"
dmodel.load_weights(model_save_loc)


yhat = dmodel(val_input)
assert isinstance(yhat, tfd.Distribution)


yhats = [dmodel(val_input) for _ in range(100)]
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
plt.title(f'{name} Predictions with Uncertainties')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.xlim((val_dates[50][0],val_dates[350][0]))
plt.show()






#testing
test_size = -80
test_input = np.lib.stride_tricks.sliding_window_view(df.iloc[test_size:-1,1].values, sequence_length)
y3 = np.array(df.iloc[sequence_length+test_size:,1]).T.astype(np.float32).reshape(-1,1)
y3shift = np.roll(y3, shift=1)
y3shift[0] = np.nan
test_dates = np.array(df.iloc[sequence_length+test_size:,0]).T.reshape(-1,1)

yhats_test = [dmodel(test_input) for _ in range(100)]
for i, yhat in enumerate(yhats_test):
  m_test = np.squeeze(yhat.mean())
  s_test = np.squeeze(yhat.stddev())

print(test_input.shape,y3.shape,test_dates.shape,m_test.shape)
upper_bound = m_test + s_test
lower_bound = m_test - s_test

# plt.figure(figsize=(10,5))
# plt.plot(test_dates, y3, label='True Values', color='black',marker='o',markersize=3)
# plt.plot(test_dates, y3shift, label='Shift Values', color='grey',marker='o',ms=3)
# plt.plot(test_dates, m_test, label='Predicted Mean', color='red',marker='x',markersize=3)
# plt.fill_between(test_dates.reshape(-1), lower_bound, upper_bound, color='orange', alpha=0.3, label='Uncertainty Interval')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.title(f'{name} Predictions with Uncertainties')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# plt.show()




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


mape = calculate_mape(y3, m_test)
rmse = calculate_rmse(y3, m_test)
mae = calculate_mae(y3, m_test)
fb = calculate_forecast_bias(y3, m_test)
r2 = calculate_r2(y3, m_test)
print("MAPE:", mape,'\nRMSE:',rmse,'\nMAE:',mae,'\nFB:',fb,'\nR2:',r2)
print()
mape = calculate_mape(y3, np.nan_to_num(y3shift,nan=0.0))
rmse = calculate_rmse(y3, np.nan_to_num(y3shift,nan=0.0))
mae = calculate_mae(y3, np.nan_to_num(y3shift,nan=0.0))
fb = calculate_forecast_bias(y3, np.nan_to_num(y3shift,nan=0.0))
r2 = calculate_r2(y3, np.nan_to_num(y3shift,nan=0.0))
print("MAPE:", mape,'\nRMSE:',rmse,'\nMAE:',mae,'\nFB:',fb,'\nR2:',r2)

y2 = y2.reshape(-1)
print(np.diff(y2).shape,np.diff(y2).tolist())
print(y2.shape)
true_pct_changes = (np.diff(y2) / y2[:-1]) * 100
pred_pct_changes = ((m - y2) / y2) * 100 
# plt.figure()
# plt.scatter(true_pct_changes, pred_pct_changes[:-1],color='orange',alpha=.4,label='Pred - Prev. Truth')
# plt.plot(true_pct_changes, true_pct_changes,lw=4,color='cadetblue')
# plt.axhline(y=0,ls='--',color='red')
# plt.axvline(x=0,ls='--',color='red')
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.legend()
# # plt.show()
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(np.sign(true_pct_changes), np.sign(pred_pct_changes[:-1]), labels=[1, -1])
print("Confusion Matrix:")
print(conf_matrix)

same_sign_indices = np.where(np.sign(true_pct_changes) == np.sign(pred_pct_changes[:-1]))
diff_sign_indices = np.where(np.sign(true_pct_changes) != np.sign(pred_pct_changes[:-1]))
# plt.figure(figsize=(6,4))
# _, bins, _ = plt.hist(true_pct_changes,bins=50,histtype='step',label='Actual Daily Returns')
# plt.hist(true_pct_changes[same_sign_indices],bins=bins,histtype='step',color='green',label='Correct')
# plt.hist(true_pct_changes[diff_sign_indices],bins=bins,histtype='step',color='red',label='Incorrect')
# plt.grid(color='0.95')
# plt.xlabel('Returns [%]')
# plt.title('Test set')
# x_min, x_max = plt.xlim()
# y_min, y_max = plt.ylim()
# text_x = x_min + 0.025 * (x_max - x_min)
# text_y = y_max - 0.1 * (y_max - y_min)
# plt.text(text_x, text_y, '% correct sign {:.3f}\n% incorrect sign {:.3f}'.format(len(true_pct_changes[same_sign_indices])/len(true_pct_changes),
#                                                                                  len(true_pct_changes[diff_sign_indices])/len(true_pct_changes)), ha='left', va='top')
# plt.legend()
# # plt.show()








#strategy
#find the days were the price today is below -2sigma or above +2sigma
print(val_dates.shape,y2.shape,m.shape,s.shape)
prediction_df = pd.DataFrame({'Date':val_dates.squeeze(),'Close':y2.squeeze(), 
                              'Model_mean':m, 'Model_sigma':s})
prediction_df['m+3s'] = prediction_df['Model_mean'] + 3*prediction_df['Model_sigma']
prediction_df['m-3s'] = prediction_df['Model_mean'] - 3*prediction_df['Model_sigma']
prediction_df.head()
print()

condition1 = prediction_df['Close'] > prediction_df['m+3s']
condition2 = prediction_df['Close'] < prediction_df['m-3s']
# combined_condition = condition1 | condition2
# ab_dates = prediction_df[combined_condition]['Date'].tolist()

above_result = prediction_df[condition1]
above_result_dates = above_result['Date'].tolist()
below_result = prediction_df[condition2]
below_result_dates = below_result['Date'].tolist()

plt.figure(figsize=(10, 6))
plt.plot(prediction_df['Date'],prediction_df['Close'], label='True Values', color='black')
plt.plot(prediction_df['Date'],prediction_df['Model_mean'], label='Predicted Mean', color='red')
plt.fill_between(prediction_df['Date'], prediction_df['m-3s'], prediction_df['m+3s'], color='indigo', alpha=0.3, label='$3\sigma$ Uncertainty')
plt.axvline(above_result_dates[0], color='g', linestyle='--', label='Above Cond. Met',alpha=0.3)
plt.axvline(below_result_dates[0], color='r', linestyle='--', label='Below Cond. Met',alpha=0.3)


combined_x_values = above_result_dates + below_result_dates
color_indicator = [1] * len(above_result_dates) + [-1] * len(below_result_dates)
combined_x_values, color_indicator = zip(*sorted(zip(combined_x_values, color_indicator)))
current_color = None  # Initialize with None
for x, color_code in zip(combined_x_values, color_indicator):
    color = 'green' if color_code == 1 else 'red'
    plt.vlines(x, ymin=0, ymax=max(prediction_df['Close']), color=color, lw=0.5,ls='--',alpha=0.3)
    if current_color is not None:
        plt.fill_betweenx([0, max(prediction_df['Close'])], previous_x, x, color=current_color, alpha=0.2) 
    current_color = color
    previous_x = x


plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title(f'{name} Predictions with Uncertainties')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.xlim((prediction_df[50][0],prediction_df[350][0]))
plt.show()





