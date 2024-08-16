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
train_size = len(df) - 350
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



negloglik = lambda y, rv_y: -rv_y.log_prob(y) #-y_pred.log_prob(y_true)
deep_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss=negloglik)
history_ = deep_model.fit(train_input, y_train, epochs=750, verbose=1)

model_folder = '/Users/leonbozianu/work/lightbox/models/{}'.format("model-5-mins")
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
deep_model.save_weights(filepath=model_folder+'/weights-5-mins.h5')
print("Model weights saved")


plt.figure(figsize=(4,3))

plt.plot(history_.history['loss'], label='Train loss B')
# plt.plot(history_.history['val_loss'], label='Val loss B')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small')
plt.tight_layout()
plt.show()


print('TRAIN UNTIL:',df.iloc[train_size],'\tTEST UNTIL:',df.iloc[-1],'\n\n\n')
yhat = deep_model(val_input)
assert isinstance(yhat, tfd.Distribution)
