import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas_ta

import tensorflow as tf
import yfinance as yf

from utils import prepare_dataframe, build_lstm_model


name = 'SPY'
train_end_date = '2022-01-01'

#download stock data the model has not seen (test set)
forecast_df = yf.download(name, start=train_end_date, end='2022-08-01')
forecast_df = forecast_df.reset_index().rename(columns={'index': 'Date'})

n_past = 20 #this must be the same as in the model training
n_future = 1 #this must be the same as in the model training
forecast_date_array = forecast_df['Date'].to_numpy()[n_past:]
truth_opens = forecast_df['Open'].values[n_past:]
truth_closes = forecast_df['Close'].values[n_past:]

columns_to_use = ['Open', 'High', 'Low', 'Close', 'Volume']
testX, testY, scaler = prepare_dataframe(forecast_df,columns_to_use,n_past,n_future)




#load the model and use it to predict the next Close
model = build_lstm_model(input_shape=(n_past, len(columns_to_use)))
# Restore the weights
model_dir = "/Users/leonbozianu/work/lightbox/models/LSTM3DO_20past_1fut_15e_14982samples"
model_save_loc = model_dir + "/final_weight.h5"
model.load_weights(model_save_loc)

forecast = model.predict(testX)
y_pred_model = scaler['Close'].inverse_transform(forecast).squeeze()

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
fig.savefig(model_dir+"/test_set_Close.png")



prediction_df['pct_change'] = prediction_df['Close'].pct_change().fillna(0)
prediction_df['pct_change_pred'] = (prediction_df['Close_pred'] - prediction_df['Close']) / prediction_df['Close']
# prediction_df['pct_change_pred'] = (prediction_df['Close_pred'] - prediction_df['Close_pred'].shift(-1)) / prediction_df['Close_pred'].shift(-1)
prediction_df['pct_change_sign'] = prediction_df['pct_change'].apply(lambda x: 1 if x > 0 else -1)
prediction_df['pred_truth_pct_change_sign'] = prediction_df['pct_change_pred'].apply(lambda x: 1 if x > 0 else -1)
prediction_df['prediction_correct'] = prediction_df['pct_change_sign'] == prediction_df['pred_truth_pct_change_sign']

print(prediction_df.head())
plt.figure()
plt.scatter(prediction_df['pct_change'], prediction_df['pct_change_pred'],color='orange',alpha=.4)
plt.plot(prediction_df['pct_change'], prediction_df['pct_change'],lw=4,color='cadetblue')
plt.axhline(y=0,ls='--',color='red')
plt.axvline(x=0,ls='--',color='red')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.savefig(model_dir+"/test_set_scatter.png")



plt.figure(figsize=(6,4))
# _, bins, _ = plt.hist(prediction_df['pct_change_pred'],bins=50,histtype='step',color='red',label='Pred')
# _, bins, _ = plt.hist(prediction_df['pct_change'],bins=50,histtype='step',color='black',label='Truth')
plt.hist(prediction_df['pct_change_pred'].values[prediction_df['pct_change'].values > 0],bins=20,histtype='step',color='green',label='Pred (Tru up)')
plt.hist(prediction_df['pct_change_pred'].values[prediction_df['pct_change'].values < 0],bins=20,histtype='step',color='red',label='Pred (Tru down)')
plt.grid(color='0.95')
plt.xlabel('pct change')
plt.legend()
plt.savefig(model_dir+'/test_set_hist.png')


conf_matrix = confusion_matrix(prediction_df['pct_change_sign'], prediction_df['pred_truth_pct_change_sign'], labels=[1, -1])
print("Confusion Matrix:")
print(conf_matrix)



print(prediction_df.head())


plt.figure()
plt.hist(prediction_df['pct_change'].values[prediction_df['prediction_correct']==True],bins=50,histtype='step',color='green',label='Correct')
plt.hist(prediction_df['pct_change'].values[prediction_df['prediction_correct']==False],bins=50,histtype='step',color='red',label='Incorrect')
plt.title('Returns (correct vs incorrect sign)')
plt.xlabel('pct change')
plt.ylabel('Freq.')
plt.grid(color="0.95")
plt.savefig(model_dir+"/test_set_when_correct.png")