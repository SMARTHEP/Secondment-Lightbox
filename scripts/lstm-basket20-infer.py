from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import pandas_ta
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, RobustScaler
import yfinance as yf




def get_all_indicators(datafr):
    datafr['1d_close_pct'] = datafr['Close'].pct_change(1)
    datafr['3d_close_pct'] = datafr['Close'].pct_change(3)
    datafr['5d_close_pct'] = datafr['Close'].pct_change(5)
    datafr['10d_close_pct'] = datafr['Close'].pct_change(10)

    datafr['10d_volatility'] = datafr['Close'].rolling(window=10).std()
    tr_df = pd.DataFrame()
    tr_df['High-Low'] = datafr['High'] - datafr['Low']
    tr_df['High-ClosePrev'] = abs(datafr['High'] - datafr['Close'].shift(1))
    tr_df['Low-ClosePrev'] = abs(datafr['Low'] - datafr['Close'].shift(1))
    datafr['TR'] = tr_df[['High-Low', 'High-ClosePrev', 'Low-ClosePrev']].max(axis=1)
    datafr['ATR10'] = datafr['TR'].rolling(window=10).mean()
    datafr['WATR'] = datafr['TR'].rolling(window=14).apply(lambda x: np.average(x, weights=np.arange(1, 14 + 1))) # Weighted Average True Range (WATR)

    log_returns = np.log(datafr['Close']/datafr['Close'].shift(1))
    datafr['VOL'] = log_returns.fillna(0).rolling(window=10).std()*np.sqrt(10) #assuming 252 trading days a year #now we just care about 10
    datafr['VS'] = np.log(datafr['High']) - np.log(datafr['Low'])
    datafr['VP'] = 0.361 * (np.log(datafr['High']/datafr['Low']))**2
    datafr['VGK'] = 0.5 * (np.log(datafr['High'])-np.log(datafr['Low']))**2 - (2*np.log(2)-1)*(np.log(datafr['Close'])-np.log(datafr['Open']))**2
    datafr['VRS'] = (np.log(datafr['High'])-np.log(datafr['Open']))*(np.log(datafr['High'])-np.log(datafr['Close'])) - (np.log(datafr['Low'])-np.log(datafr['Open']))*(np.log(datafr['Low'])-np.log(datafr['Close']))

    datafr['SMA20'] = datafr.Close.rolling(20).mean()
    datafr['EMA20'] = datafr.Close.ewm(span=20).mean()
    datafr['SMA100'] = datafr.Close.rolling(100).mean()
    datafr['EMA100'] = datafr.Close.ewm(span=100).mean()

    datafr['Open_SMA50_diff'] = datafr['Open'].sub(datafr['SMA20'])
    datafr['Open_EMA20_diff'] = datafr['Open'].sub(datafr['EMA20'])
    datafr['Open_SMA100_diff'] = datafr['Open'].sub(datafr['SMA100'])
    datafr['Open_EMA100_diff'] = datafr['Open'].sub(datafr['EMA100'])

    #pandas ta library
    datafr['RSI_14'] = pandas_ta.rsi(datafr['Close'],length=14)
    datafr['RSI_100'] = pandas_ta.rsi(datafr['Close'],length=100)

    vor_df = pandas_ta.vortex(datafr['High'],datafr['Low'],datafr['Close'],14)
    datafr['VTXP_14'] = vor_df['VTXP_14']
    datafr['VTXM_14'] = vor_df['VTXM_14']
    stoch_df = pandas_ta.stoch(datafr['High'],datafr['Low'],datafr['Close'],k=14,d=26)
    datafr['STOCHk_14_26_3'] = stoch_df['STOCHk_14_26_3']
    datafr['STOCHd_14_26_3'] = stoch_df['STOCHd_14_26_3']
    adx_df = pandas_ta.adx(datafr['High'],datafr['Low'],datafr['Close'],length=14)#average directional index
    datafr['ADX_14'] = adx_df['ADX_14']
    datafr['DMP_14'] = adx_df['DMP_14']
    datafr['DMN_14'] = adx_df['DMN_14']
    amat_df = pandas_ta.amat(datafr['Close']) #Archer Moving Averages Trends
    datafr['AMATe_LR_8_21_2'] = amat_df['AMATe_LR_8_21_2']
    datafr['AMATe_SR_8_21_2'] = amat_df['AMATe_SR_8_21_2']
    aroon_df = pandas_ta.aroon(datafr['High'],datafr['Low']) #Aroon & Aroon Oscillator
    datafr['AROOND_14'] = aroon_df['AROOND_14']
    datafr['AROONU_14'] = aroon_df['AROONU_14']
    datafr['AROONOSC_14'] = aroon_df['AROONOSC_14']
    ao_df = pandas_ta.ao(datafr['High'],datafr['Low'])#Awesome Oscillator
    datafr['AO_5_34'] = ao_df
    uo_df = pandas_ta.uo(datafr['High'],datafr['Low'],datafr['Close'])#Ultimate Oscillator
    datafr['UO_7_14_28'] = uo_df
    cg_df = pandas_ta.cg(datafr['Close'])#Center of Gravity
    datafr['CG_10'] = cg_df
    coppock_df = pandas_ta.coppock(datafr['Close'])#Coppock
    datafr['COPC_11_14_10'] = coppock_df
    inertia_df = pandas_ta.inertia(datafr['Close'],datafr['High'],datafr['Low'])#inertia
    datafr['INERTIA_20_14'] = inertia_df
    stc_df = pandas_ta.stc(datafr['Close']) #schaff trend cycle
    datafr['STC_10_12_26_0.5'] = stc_df['STC_10_12_26_0.5']
    datafr['STCmacd_10_12_26_0.5'] = stc_df['STCmacd_10_12_26_0.5']
    datafr['STCstoch_10_12_26_0.5'] = stc_df['STCstoch_10_12_26_0.5']
    tsi_df = pandas_ta.tsi(datafr['Close']) # true strength index
    datafr['TSI_13_25_13'] = tsi_df['TSI_13_25_13']
    qstick_df = pandas_ta.qstick(datafr['Open'],datafr['Close']) #q stick indicator
    datafr['QS_10'] = qstick_df
    vhf_df = pandas_ta.vhf(datafr['Close'])#vertical horizontal filter
    datafr['VHF_28'] = vhf_df
    dpo_df = pandas_ta.dpo(datafr['Close']) #detrend price oscillator
    datafr['DPO_20'] = dpo_df
    pdist_df = pandas_ta.pdist(datafr['Open'],datafr['High'],datafr['Low'],datafr['Close'])
    datafr['PDIST'] = pdist_df
    rvi_df = pandas_ta.rvi(datafr['Close'],datafr['High'],datafr['Low']) #relative volatility index
    datafr['RVI_14'] = rvi_df
    willr_df = pandas_ta.willr(datafr['High'],datafr['Low'],datafr['Close']) # William's Percent R (WILLR)
    datafr['WILLR_14'] = willr_df
    ebsw_df = pandas_ta.ebsw(datafr['Close']) #Even Better SineWave (EBSW)
    datafr['EBSW_40_10'] = ebsw_df
    kurt_df = pandas_ta.kurtosis(datafr['Close']) #kurtosis
    datafr['KURT_30'] = kurt_df
    zscore_df = pandas_ta.zscore(datafr['Close'])
    datafr['ZS_30'] = zscore_df
    chop_df = pandas_ta.chop(datafr['High'],datafr['Low'],datafr['Close']) #choppiness index
    datafr['CHOP_14_1_100'] = chop_df

    print(datafr.shape)
    datafr = datafr.dropna()
    print(datafr.shape)

    return datafr






def build_lstm_model(input_shape):

    inputs = tf.keras.layers.Input(shape=input_shape)
    lstm_output1 = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    dropout_output1 = tf.keras.layers.Dropout(0.3)(lstm_output1)  # Adding dropout with a rate of 0.2
    lstm_output2 = tf.keras.layers.LSTM(128, return_sequences=True)(dropout_output1)
    dropout_output2 = tf.keras.layers.Dropout(0.3)(lstm_output2)  
    lstm_output3 = tf.keras.layers.LSTM(128, return_sequences=False)(dropout_output2)
    dropout_output3 = tf.keras.layers.Dropout(0.3)(lstm_output3)  

    flat_mean_output = tf.keras.layers.Dense(5, activation='relu')(dropout_output3) 
    output_mean = tf.keras.layers.Dense(1)(flat_mean_output)  # Output for mean prediction

    model = tf.keras.models.Model(inputs=inputs, outputs=output_mean)

    return model


def prepare_dataframe(df,cols_to_use,close_col_idx=3):
    updated_df = df.copy()

    updated_df = updated_df[cols_to_use] 
    upd_scaler = StandardScaler()
    upd_scaler = upd_scaler.fit(updated_df)
    updated_df_scaled = upd_scaler.transform(updated_df)

    X = []
    Y = []
    n_future = 1   # Number of days we want to look into the future based on the past days.
    n_past = 14  # Number of past days we want to use to predict the future.
    for i in range(n_past, len(updated_df_scaled) - n_future + 1,n_future):
        X.append(updated_df_scaled[i - n_past:i, 0:updated_df_scaled.shape[1]])
        Y.append(updated_df_scaled[i + n_future - 1:i + n_future, close_col_idx]) #index 0 is Open, 3 is Close
        # Y.append(updated_df_scaled[i:i + n_future, 0]) #index 0 is Open

    return np.array(X), np.array(Y), upd_scaler








name = 'JPM'
train_end_date = '2022-01-01'

# train_end_date = '2017-12-31'
print(train_end_date)
forecast_df = yf.download(name, start=train_end_date, end='2023-07-01')
forecast_df = forecast_df.reset_index().rename(columns={'index': 'Date'})
forecast_df = get_all_indicators(forecast_df)






n_past = 14
forecast_date_array = forecast_df['Date'].to_numpy()[n_past:]
truth_opens = forecast_df['Open'].values[n_past:]
truth_closes = forecast_df['Close'].values[n_past:]
truth_sma = forecast_df['SMA20'].values[n_past:]
truth_vs = forecast_df['TR'].values[n_past:]


columns_to_use_1 = ['Close']
columns_to_use_2 = ['Close', 'ATR10']
columns_to_use_5 = ['Open', 'High', 'Low', 'Close', 'Volume']
columns_to_use_7 = ['Close', '1d_close_pct', 'ATR10', 'VOL', 'EMA20', 'PDIST', 'ADX_14']
columns_to_use_max = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', '1d_close_pct',
                        '3d_close_pct', '5d_close_pct', '10d_close_pct', '10d_volatility', 'TR',
                        'ATR10', 'WATR', 'VOL', 'VS', 'VP', 'VGK', 'VRS', 'SMA20', 'EMA20', 
                        'SMA100', 'EMA100','Open_EMA20_diff', 'Open_SMA100_diff',
                        'Open_EMA100_diff', 'RSI_14', 'RSI_100', 'VTXP_14', 'VTXM_14',
                        'STOCHk_14_26_3', 'STOCHd_14_26_3', 'ADX_14', 'DMP_14', 'DMN_14',
                        'AMATe_LR_8_21_2', 'AMATe_SR_8_21_2', 'AROOND_14', 'AROONU_14',
                        'AROONOSC_14', 'AO_5_34',
                        'UO_7_14_28', 'CG_10', 'COPC_11_14_10', 'INERTIA_20_14',
                        'STC_10_12_26_0.5', 'STCmacd_10_12_26_0.5', 'STCstoch_10_12_26_0.5',
                        'TSI_13_25_13', 'QS_10', 'VHF_28', 'DPO_20', 'PDIST', 'RVI_14',
                        'WILLR_14', 'EBSW_40_10', 'KURT_30', 'ZS_30', 'CHOP_14_1_100']

testX_1, testY_1, scaler_1 = prepare_dataframe(forecast_df,columns_to_use_1,0)
testX_2, testY_2, scaler_2 = prepare_dataframe(forecast_df,columns_to_use_2,0)
testX_5, testY_5, scaler_5 = prepare_dataframe(forecast_df,columns_to_use_5)
testX_7, testY_7, scaler_7 = prepare_dataframe(forecast_df,columns_to_use_7,0)
testX_max, testY_max, scaler_max = prepare_dataframe(forecast_df,columns_to_use_max)


forecast_date_array = forecast_df['Date'].to_numpy()[n_past:]
truth_closes = forecast_df['Close'].values[n_past:]
truth_sma = forecast_df['SMA20'].values[n_past:]
truth_vs = forecast_df['TR'].values[n_past:]


model_1 = build_lstm_model(input_shape=(n_past, len(columns_to_use_1)))
# Restore the weights
model_save_loc = "/Users/leonbozianu/work/lightbox/models/model_1" + "/final_weight.h5"
model_1.load_weights(model_save_loc)
#predict using the Close column only
forecast_1 = model_1.predict(testX_1)
y_pred_1 = scaler_1.inverse_transform(forecast_1)[:,0]



model_2 = build_lstm_model(input_shape=(n_past, len(columns_to_use_2)))
# Restore the weights
model_2.load_weights("/Users/leonbozianu/work/lightbox/models/model_2/final_weight.h5")
#predict using the Close, ATR10 columns
forecast_2 = model_2.predict(testX_2)
copy_2 = np.repeat(forecast_2,len(columns_to_use_2),axis=-1)
y_pred_2 = scaler_2.inverse_transform(copy_2)[:,0]


model_5 = build_lstm_model(input_shape=(n_past, len(columns_to_use_5)))
# Restore the weights
model_5.load_weights("/Users/leonbozianu/work/lightbox/models/model_5/final_weight.h5")
#predict using the OHLCV columns
forecast_5 = model_5.predict(testX_5)
copy_5 = np.repeat(forecast_5,len(columns_to_use_5),axis=-1)
y_pred_5 = scaler_5.inverse_transform(copy_5)[:,3]

model_7 = build_lstm_model(input_shape=(n_past, len(columns_to_use_7)))
# Restore the weights
model_7.load_weights("/Users/leonbozianu/work/lightbox/models/model_7/final_weight.h5")
#predict using the 7 columns
forecast_7 = model_7.predict(testX_7)
copy_7 = np.repeat(forecast_7,len(columns_to_use_7),axis=-1)
y_pred_7 = scaler_7.inverse_transform(copy_7)[:,0]


model_max = build_lstm_model(input_shape=(n_past, len(columns_to_use_max)))
# Restore the weights
model_max.load_weights("/Users/leonbozianu/work/lightbox/models/model_max/final_weight.h5")
#predict using the 7 columns
forecast_max = model_max.predict(testX_max)
copy_max = np.repeat(forecast_max,len(columns_to_use_max),axis=-1)
y_pred_max = scaler_max.inverse_transform(copy_max)[:,3]


print(y_pred_1.shape,y_pred_2.shape,y_pred_5.shape,y_pred_7.shape,y_pred_max.shape)

prediction_df = pd.DataFrame({'Date':forecast_date_array,'Close':truth_closes, 'SMA20':truth_sma,
                              'Close_pred_1':y_pred_1, 'Close_pred_2':y_pred_2,
                              'Close_pred_5':y_pred_5,'Close_pred_7':y_pred_7, 
                              'Close_pred_max':y_pred_max})



fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

ax[0].plot(prediction_df.Date,prediction_df.Close,label='Close',color='black')
ax[0].plot(prediction_df.Date,prediction_df.SMA20,ls='-.',label='SMA20',color='grey')
ax[0].plot(prediction_df.Date,prediction_df.Close_pred_1,label=f'Close Pred 1',color='red')
ax[0].plot(prediction_df.Date,prediction_df.Close_pred_2,label=f'Close Pred 2',color='orange')
ax[0].plot(prediction_df.Date,prediction_df.Close_pred_5,label=f'Close Pred 5',color='purple')
ax[0].plot(prediction_df.Date,prediction_df.Close_pred_7,label=f'Close Pred 7',color='pink')
ax[0].plot(prediction_df.Date,prediction_df.Close_pred_max,label=f'Close Pred Max',color='turquoise')
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[0].set_ylabel('price')
ax[0].set_title(name +' Stock (Test Set) - Transfer Learning (20 Stock Basket)')
ax[0].grid(color="0.95")

ax[1].plot(prediction_df.Date, (prediction_df.SMA20 - prediction_df.Close)/prediction_df.Close, color='grey',label='SMA20',ls='-.')
ax[1].plot(prediction_df.Date, (prediction_df.Close_pred_1- prediction_df.Close)/prediction_df.Close,label=f'Close Pred 1',lw=0.75,color='red')
ax[1].plot(prediction_df.Date, (prediction_df.Close_pred_2- prediction_df.Close)/prediction_df.Close,label=f'Close Pred 2',lw=0.75,color='orange')
ax[1].plot(prediction_df.Date, (prediction_df.Close_pred_5- prediction_df.Close)/prediction_df.Close,label=f'Close Pred OHLCV',lw=0.75,color='purple')
ax[1].plot(prediction_df.Date, (prediction_df.Close_pred_7 - prediction_df.Close)/prediction_df.Close,label='Close Pred 7',lw=0.75,color='pink')
ax[1].plot(prediction_df.Date, (prediction_df.Close_pred_max - prediction_df.Close)/prediction_df.Close,label='Close Pred Max',lw=0.75,color='turquoise')
ax[1].axhline(0,ls='--',color='black')
ax[1].set_ylabel('$\Delta$%\n(Pred - Close)/Close')
ax[1].set_xlabel('Date')
ax[1].tick_params(axis='x', labelrotation=45)
ax[1].grid(color="0.95")
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small')
# ax[1].set_xlim(prediction_df.Date.iloc[470],prediction_df.Date.iloc[-1])
# ax[0].set_ylim(110,185)
plt.tight_layout()
plt.show()




print(len(prediction_df))
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
ax[0].plot(prediction_df.Date,prediction_df.Close,label='Close',color='black')
ax[0].plot(prediction_df.Date,prediction_df.SMA20,ls='-.',label='SMA20',color='grey')
ax[0].plot(prediction_df.Date,prediction_df.Close_pred_1,label=f'Close Pred 1',color='red',marker='x',ms=3)
ax[0].plot(prediction_df.Date,prediction_df.Close_pred_2,label=f'Close Pred 2',color='orange',marker='x',ms=3)
ax[0].plot(prediction_df.Date,prediction_df.Close_pred_5,label=f'Close Pred 5',color='purple',marker='x',ms=3)
ax[0].plot(prediction_df.Date,prediction_df.Close_pred_7,label=f'Close Pred 7',color='pink',marker='x',ms=3)
ax[0].plot(prediction_df.Date,prediction_df.Close_pred_max,label=f'Close Pred Max',color='turquoise',marker='x',ms=3)
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[0].set_ylabel('price')
ax[0].set_title(name +' Stock (Test Set) - Transfer Learning (20 Stock Basket)')
ax[0].grid()

ax[1].plot(prediction_df.Date, (prediction_df.SMA20 - prediction_df.Close)/prediction_df.Close, color='grey',label='SMA20',ls='-.')
ax[1].plot(prediction_df.Date, (prediction_df.Close_pred_1- prediction_df.Close)/prediction_df.Close,label=f'Close Pred 1',lw=0.75,color='red')
ax[1].plot(prediction_df.Date, (prediction_df.Close_pred_2- prediction_df.Close)/prediction_df.Close,label=f'Close Pred 2',lw=0.75,color='orange')
ax[1].plot(prediction_df.Date, (prediction_df.Close_pred_5- prediction_df.Close)/prediction_df.Close,label=f'Close Pred OHLCV',lw=0.75,color='purple')
ax[1].plot(prediction_df.Date, (prediction_df.Close_pred_7 - prediction_df.Close)/prediction_df.Close,label='Close Pred 7',lw=0.75,color='pink')
ax[1].plot(prediction_df.Date, (prediction_df.Close_pred_max - prediction_df.Close)/prediction_df.Close,label='Close Pred Max',lw=0.75,color='turquoise')
ax[1].axhline(0,ls='--',color='black')
ax[1].set_ylabel('$\Delta$%\n(Pred - Close)/Close')
ax[1].set_xlabel('Date')
ax[1].tick_params(axis='x', labelrotation=45)
ax[1].grid()
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small')
ax[1].set_xlim(prediction_df.Date.iloc[150],prediction_df.Date.iloc[-1])
ax[0].set_ylim(135,185)
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


truths = prediction_df.Close.shift(-1).values[:-1]
preds_1 = prediction_df.Close_pred_1.values[:-1]
preds_2 = prediction_df.Close_pred_2.values[:-1]
preds_5 = prediction_df.Close_pred_5.values[:-1]
preds_7 = prediction_df.Close_pred_7.values[:-1]
preds_max = prediction_df.Close_pred_max.values[:-1]



models = {
    "Close Column": preds_1,
    "2 Columns": preds_2,
    "5 Columns": preds_5,
    "7 Columns": preds_7,
    "All Columns": preds_max
}

metrics = ["MAPE", "RMSE", "MAE", "FB", "R2"]


print("{:<15}".format("Model"), end="")
for metric in metrics:
    print("{:<10}".format(metric), end="")
print()

for model_name, predictions in models.items():
    print("{:<15}".format(model_name), end="")
    for metric in metrics:
        metric_value = None
        if metric == "MAPE":
            metric_value = calculate_mape(predictions, truths)
        elif metric == "RMSE":
            metric_value = calculate_rmse(predictions, truths)
        elif metric == "MAE":
            metric_value = calculate_mae(predictions, truths)
        elif metric == "FB":
            metric_value = calculate_forecast_bias(predictions, truths)
        elif metric == "R2":
            metric_value = calculate_r2(predictions, truths)
        print("{:<10.2f}".format(metric_value), end="")
    print()
