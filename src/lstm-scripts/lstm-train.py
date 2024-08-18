import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yfinance as yf
import os
import argparse

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions



parser = argparse.ArgumentParser(description="Process input arguments for model training.")

parser.add_argument('--name', required=True, help="Name of the underlying")
parser.add_argument('--end_date', required=True, help="End date for the entire dataset")
parser.add_argument('--seq_length', type=int, required=True, help="Sequence length for the model")
parser.add_argument('--epochs', type=int, required=True, help="Number of training epochs")
args = parser.parse_args()


#
name = args.name
data_end_date = args.end_date
name_ohlc_df = yf.download(name, start='2004-01-01', end=data_end_date)
name_ohlc_df = name_ohlc_df.reset_index()
data = {'date': name_ohlc_df.Date, 'close': name_ohlc_df.Close, 'volume': name_ohlc_df.Volume}
df = pd.DataFrame(data)

train_size = int(0.9 * len(df))
print('TRAIN UNTIL:',df.iloc[train_size],'\tTEST UNTIL:',df.iloc[-1],'\n\n\n')

sequence_length = args.seq_length
train_input = np.lib.stride_tricks.sliding_window_view(df.iloc[:train_size,1].values, (sequence_length,))
val_input = np.lib.stride_tricks.sliding_window_view(df.iloc[train_size:-1,1].values, (sequence_length,))

y1 = np.array(df.iloc[sequence_length:train_size+1,1]).T.astype(np.float32).reshape(-1,1)
y2 = np.array(df.iloc[sequence_length+train_size:,1]).T.astype(np.float32).reshape(-1,1)

deep_model = tf.keras.Sequential([
  tf.keras.layers.LSTM(32,activation='relu',input_shape=(sequence_length,1),return_sequences=False),
  tf.keras.layers.Dense(64,activation='relu'),
  tf.keras.layers.Dense(24,activation='relu'),
  tf.keras.layers.Dense(8,activation='relu'),
  tf.keras.layers.Dense(1+1),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Normal(loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.001*t[...,1:]))),
      # lambda t: tfd.StudentT(df=4,loc=t[..., :1],scale=1e-3 + tf.math.softplus(t[...,1:])))
])


negloglik = lambda y, rv_y: -rv_y.log_prob(y) #-y_pred.log_prob(y_true)
deep_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss=negloglik)
history_ = deep_model.fit(train_input, y1, epochs=args.epochs, verbose=1)

model_folder = './models/{}'.format("model-{}".format(data_end_date))
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
deep_model.save_weights(filepath=model_folder+'/weights-end-date-{}.h5'.format(data_end_date))
print("Model weights saved")



plt.figure(figsize=(4,3))
plt.plot(history_.history['loss'], label='Train loss')
plt.plot(history_.history['val_loss'], label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small')
plt.tight_layout()
plt.show()

print('TRAIN UNTIL:',df.iloc[train_size],'\tTEST UNTIL:',df.iloc[-1],'\n\n\n')
yhat = deep_model(val_input)
assert isinstance(yhat, tfd.Distribution)
