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


name = 'AAPL'
train_end_date = '2022-01-01'


_ohlc_df = yf.download(name, start='2012-01-01', end=train_end_date)
xom_ohlc_df = yf.download('XOM', start='2012-01-01', end=train_end_date)
shw_ohlc_df = yf.download('SHW', start='2012-01-01', end=train_end_date)
unp_ohlc_df = yf.download('UNP', start='2012-01-01', end=train_end_date)
nee_ohlc_df = yf.download('NEE', start='2012-01-01', end=train_end_date)
pfe_ohlc_df = yf.download('PFE', start='2012-01-01', end=train_end_date)
jpm_ohlc_df = yf.download('JPM', start='2012-01-01', end=train_end_date)
amzn_ohlc_df = yf.download('AMZN', start='2012-01-01', end=train_end_date)
pg_ohlc_df = yf.download('PG', start='2012-01-01', end=train_end_date)
aapl_ohlc_df = yf.download('AAPL', start='2012-01-01', end=train_end_date)
goog_ohlc_df = yf.download('GOOG', start='2012-01-01', end=train_end_date)

cvx_ohlc_df = yf.download('CVX', start='2012-01-01', end=train_end_date)
nem_ohlc_df = yf.download('NEM', start='2012-01-01', end=train_end_date)
rtx_ohlc_df = yf.download('RTX', start='2012-01-01', end=train_end_date)
duk_ohlc_df = yf.download('DUK', start='2012-01-01', end=train_end_date)
lly_ohlc_df = yf.download('LLY', start='2012-01-01', end=train_end_date)
wfc_ohlc_df = yf.download('WFC', start='2012-01-01', end=train_end_date)
tsla_ohlc_df = yf.download('TSLA', start='2012-01-01', end=train_end_date)
ko_ohlc_df = yf.download('KO', start='2012-01-01', end=train_end_date)
msft_ohlc_df = yf.download('MSFT', start='2012-01-01', end=train_end_date)
dis_ohlc_df = yf.download('DIS', start='2012-01-01', end=train_end_date)



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




_ohlc_df = get_all_indicators(_ohlc_df)
xom_ohlc_df = get_all_indicators(xom_ohlc_df)
shw_ohlc_df = get_all_indicators(shw_ohlc_df)
unp_ohlc_df = get_all_indicators(unp_ohlc_df)
nee_ohlc_df = get_all_indicators(nee_ohlc_df)
pfe_ohlc_df = get_all_indicators(pfe_ohlc_df)
jpm_ohlc_df = get_all_indicators(jpm_ohlc_df)
amzn_ohlc_df = get_all_indicators(amzn_ohlc_df)
pg_ohlc_df = get_all_indicators(pg_ohlc_df)
aapl_ohlc_df = get_all_indicators(aapl_ohlc_df)
goog_ohlc_df = get_all_indicators(goog_ohlc_df)


cvx_ohlc_df = get_all_indicators(cvx_ohlc_df)
nem_ohlc_df = get_all_indicators(nem_ohlc_df)
rtx_ohlc_df = get_all_indicators(rtx_ohlc_df)
duk_ohlc_df = get_all_indicators(duk_ohlc_df)
lly_ohlc_df = get_all_indicators(lly_ohlc_df)
wfc_ohlc_df = get_all_indicators(wfc_ohlc_df)
tsla_ohlc_df = get_all_indicators(tsla_ohlc_df)
ko_ohlc_df = get_all_indicators(ko_ohlc_df)
msft_ohlc_df = get_all_indicators(msft_ohlc_df)
dis_ohlc_df = get_all_indicators(dis_ohlc_df)


def prepare_dataframe(df,cols_to_use,close_col_idx=3):
    updated_df = df.copy()

    updated_df = updated_df[cols_to_use] 
    upd_scaler = StandardScaler()
    upd_scaler = upd_scaler.fit(updated_df)
    updated_df_scaled = upd_scaler.transform(updated_df)

    X = []
    Y = []
    n_future = 1   # Number of days we want to look into the future based on the past days.
    global n_past
    n_past = 14  # Number of past days we want to use to predict the future.
    for i in range(n_past, len(updated_df_scaled) - n_future + 1,n_future):
        X.append(updated_df_scaled[i - n_past:i, 0:updated_df_scaled.shape[1]])
        Y.append(updated_df_scaled[i + n_future - 1:i + n_future, close_col_idx]) #index 0 is Open, 3 is Close
        # Y.append(updated_df_scaled[i:i + n_future, 0]) #index 0 is Open

    return np.array(X), np.array(Y), upd_scaler



cols = list(_ohlc_df)
print(cols) #['Open', 'High', 'Low', 'Close', 'Adj Clos

df_for_training1 = _ohlc_df[cols].astype(float) 
df_for_training2 = xom_ohlc_df[cols].astype(float) 
df_for_training3 = shw_ohlc_df[cols].astype(float) 
df_for_training4 = unp_ohlc_df[cols].astype(float) 
df_for_training5 = nee_ohlc_df[cols].astype(float) 
df_for_training6 = pfe_ohlc_df[cols].astype(float) 
df_for_training7 = jpm_ohlc_df[cols].astype(float) 
df_for_training8 = amzn_ohlc_df[cols].astype(float) 
df_for_training9 = pg_ohlc_df[cols].astype(float) 
df_for_training10 = aapl_ohlc_df[cols].astype(float) 
df_for_training11 = goog_ohlc_df[cols].astype(float) 

df_for_training12 = cvx_ohlc_df[cols].astype(float)
df_for_training13 = nem_ohlc_df[cols].astype(float)
df_for_training14 = rtx_ohlc_df[cols].astype(float)
df_for_training15 = duk_ohlc_df[cols].astype(float)
df_for_training16 = lly_ohlc_df[cols].astype(float)
df_for_training17 = wfc_ohlc_df[cols].astype(float)
df_for_training18 = tsla_ohlc_df[cols].astype(float)
df_for_training19 = ko_ohlc_df[cols].astype(float)
df_for_training20 = msft_ohlc_df[cols].astype(float)
df_for_training21 = dis_ohlc_df[cols].astype(float)


columns_to_use_1 = ['Close']
trainX1_1, trainY1_1, train_scaler = prepare_dataframe(df_for_training1,columns_to_use_1,0)
trainX2_1, trainY2_1, train_scaler = prepare_dataframe(df_for_training2,columns_to_use_1,0)
trainX3_1, trainY3_1, train_scaler = prepare_dataframe(df_for_training3,columns_to_use_1,0)
trainX4_1, trainY4_1, train_scaler = prepare_dataframe(df_for_training4,columns_to_use_1,0)
trainX5_1, trainY5_1, train_scaler = prepare_dataframe(df_for_training5,columns_to_use_1,0)
trainX6_1, trainY6_1, train_scaler = prepare_dataframe(df_for_training6,columns_to_use_1,0)
trainX7_1, trainY7_1, train_scaler = prepare_dataframe(df_for_training7,columns_to_use_1,0)
trainX8_1, trainY8_1, train_scaler = prepare_dataframe(df_for_training8,columns_to_use_1,0)
trainX9_1, trainY9_1, train_scaler = prepare_dataframe(df_for_training9,columns_to_use_1,0)
trainX10_1, trainY10_1, train_scaler = prepare_dataframe(df_for_training10,columns_to_use_1,0)
trainX11_1, trainY11_1, train_scaler = prepare_dataframe(df_for_training11,columns_to_use_1,0)

trainX12_1, trainY12_1, train_scaler = prepare_dataframe(df_for_training12,columns_to_use_1,0)
trainX13_1, trainY13_1, train_scaler = prepare_dataframe(df_for_training13,columns_to_use_1,0)
trainX14_1, trainY14_1, train_scaler = prepare_dataframe(df_for_training14,columns_to_use_1,0)
trainX15_1, trainY15_1, train_scaler = prepare_dataframe(df_for_training15,columns_to_use_1,0)
trainX16_1, trainY16_1, train_scaler = prepare_dataframe(df_for_training16,columns_to_use_1,0)
trainX17_1, trainY17_1, train_scaler = prepare_dataframe(df_for_training17,columns_to_use_1,0)
trainX18_1, trainY18_1, train_scaler = prepare_dataframe(df_for_training18,columns_to_use_1,0)
trainX19_1, trainY19_1, train_scaler = prepare_dataframe(df_for_training19,columns_to_use_1,0)
trainX20_1, trainY20_1, train_scaler = prepare_dataframe(df_for_training20,columns_to_use_1,0)
trainX21_1, trainY21_1, train_scaler = prepare_dataframe(df_for_training21,columns_to_use_1,0)


columns_to_use_2 = ['Close', 'ATR10']
trainX1_2, trainY1_2, train_scaler = prepare_dataframe(df_for_training1,columns_to_use_2,0)
trainX2_2, trainY2_2, train_scaler = prepare_dataframe(df_for_training2,columns_to_use_2,0)
trainX3_2, trainY3_2, train_scaler = prepare_dataframe(df_for_training3,columns_to_use_2,0)
trainX4_2, trainY4_2, train_scaler = prepare_dataframe(df_for_training4,columns_to_use_2,0)
trainX5_2, trainY5_2, train_scaler = prepare_dataframe(df_for_training5,columns_to_use_2,0)
trainX6_2, trainY6_2, train_scaler = prepare_dataframe(df_for_training6,columns_to_use_2,0)
trainX7_2, trainY7_2, train_scaler = prepare_dataframe(df_for_training7,columns_to_use_2,0)
trainX8_2, trainY8_2, train_scaler = prepare_dataframe(df_for_training8,columns_to_use_2,0)
trainX9_2, trainY9_2, train_scaler = prepare_dataframe(df_for_training9,columns_to_use_2,0)
trainX10_2, trainY10_2, train_scaler = prepare_dataframe(df_for_training10,columns_to_use_2,0)
trainX11_2, trainY11_2, train_scaler = prepare_dataframe(df_for_training11,columns_to_use_2,0)

trainX12_2, trainY12_2, train_scaler = prepare_dataframe(df_for_training12,columns_to_use_2,0)
trainX13_2, trainY13_2, train_scaler = prepare_dataframe(df_for_training13,columns_to_use_2,0)
trainX14_2, trainY14_2, train_scaler = prepare_dataframe(df_for_training14,columns_to_use_2,0)
trainX15_2, trainY15_2, train_scaler = prepare_dataframe(df_for_training15,columns_to_use_2,0)
trainX16_2, trainY16_2, train_scaler = prepare_dataframe(df_for_training16,columns_to_use_2,0)
trainX17_2, trainY17_2, train_scaler = prepare_dataframe(df_for_training17,columns_to_use_2,0)
trainX18_2, trainY18_2, train_scaler = prepare_dataframe(df_for_training18,columns_to_use_2,0)
trainX19_2, trainY19_2, train_scaler = prepare_dataframe(df_for_training19,columns_to_use_2,0)
trainX20_2, trainY20_2, train_scaler = prepare_dataframe(df_for_training20,columns_to_use_2,0)
trainX21_2, trainY21_2, train_scaler = prepare_dataframe(df_for_training21,columns_to_use_2,0)




columns_to_use_5 = ['Open', 'High', 'Low', 'Close', 'Volume']
trainX1_5, trainY1_5, train_scaler = prepare_dataframe(df_for_training1,columns_to_use_5,0)
trainX2_5, trainY2_5, train_scaler = prepare_dataframe(df_for_training2,columns_to_use_5,0)
trainX3_5, trainY3_5, train_scaler = prepare_dataframe(df_for_training3,columns_to_use_5,0)
trainX4_5, trainY4_5, train_scaler = prepare_dataframe(df_for_training4,columns_to_use_5,0)
trainX5_5, trainY5_5, train_scaler = prepare_dataframe(df_for_training5,columns_to_use_5,0)
trainX6_5, trainY6_5, train_scaler = prepare_dataframe(df_for_training6,columns_to_use_5,0)
trainX7_5, trainY7_5, train_scaler = prepare_dataframe(df_for_training7,columns_to_use_5,0)
trainX8_5, trainY8_5, train_scaler = prepare_dataframe(df_for_training8,columns_to_use_5,0)
trainX9_5, trainY9_5, train_scaler = prepare_dataframe(df_for_training9,columns_to_use_5,0)
trainX10_5, trainY10_5, train_scaler = prepare_dataframe(df_for_training10,columns_to_use_5,0)
trainX11_5, trainY11_5, train_scaler = prepare_dataframe(df_for_training11,columns_to_use_5,0)

trainX12_5, trainY12_5, train_scaler = prepare_dataframe(df_for_training12,columns_to_use_5,0)
trainX13_5, trainY13_5, train_scaler = prepare_dataframe(df_for_training13,columns_to_use_5,0)
trainX14_5, trainY14_5, train_scaler = prepare_dataframe(df_for_training14,columns_to_use_5,0)
trainX15_5, trainY15_5, train_scaler = prepare_dataframe(df_for_training15,columns_to_use_5,0)
trainX16_5, trainY16_5, train_scaler = prepare_dataframe(df_for_training16,columns_to_use_5,0)
trainX17_5, trainY17_5, train_scaler = prepare_dataframe(df_for_training17,columns_to_use_5,0)
trainX18_5, trainY18_5, train_scaler = prepare_dataframe(df_for_training18,columns_to_use_5,0)
trainX19_5, trainY19_5, train_scaler = prepare_dataframe(df_for_training19,columns_to_use_5,0)
trainX20_5, trainY20_5, train_scaler = prepare_dataframe(df_for_training20,columns_to_use_5,0)
trainX21_5, trainY21_5, train_scaler = prepare_dataframe(df_for_training21,columns_to_use_5,0)



columns_to_use_7 = ['Close', '1d_close_pct', 'ATR10', 'VOL', 'EMA20', 'PDIST', 'ADX_14']
trainX1_7, trainY1_7, train_scaler = prepare_dataframe(df_for_training1,columns_to_use_7,0)
trainX2_7, trainY2_7, train_scaler = prepare_dataframe(df_for_training2,columns_to_use_7,0)
trainX3_7, trainY3_7, train_scaler = prepare_dataframe(df_for_training3,columns_to_use_7,0)
trainX4_7, trainY4_7, train_scaler = prepare_dataframe(df_for_training4,columns_to_use_7,0)
trainX5_7, trainY5_7, train_scaler = prepare_dataframe(df_for_training5,columns_to_use_7,0)
trainX6_7, trainY6_7, train_scaler = prepare_dataframe(df_for_training6,columns_to_use_7,0)
trainX7_7, trainY7_7, train_scaler = prepare_dataframe(df_for_training7,columns_to_use_7,0)
trainX8_7, trainY8_7, train_scaler = prepare_dataframe(df_for_training8,columns_to_use_7,0)
trainX9_7, trainY9_7, train_scaler = prepare_dataframe(df_for_training9,columns_to_use_7,0)
trainX10_7, trainY10_7, train_scaler = prepare_dataframe(df_for_training10,columns_to_use_7,0)
trainX11_7, trainY11_7, train_scaler = prepare_dataframe(df_for_training11,columns_to_use_7,0)


trainX12_7, trainY12_7, train_scaler = prepare_dataframe(df_for_training12,columns_to_use_7,0)
trainX13_7, trainY13_7, train_scaler = prepare_dataframe(df_for_training13,columns_to_use_7,0)
trainX14_7, trainY14_7, train_scaler = prepare_dataframe(df_for_training14,columns_to_use_7,0)
trainX15_7, trainY15_7, train_scaler = prepare_dataframe(df_for_training15,columns_to_use_7,0)
trainX16_7, trainY16_7, train_scaler = prepare_dataframe(df_for_training16,columns_to_use_7,0)
trainX17_7, trainY17_7, train_scaler = prepare_dataframe(df_for_training17,columns_to_use_7,0)
trainX18_7, trainY18_7, train_scaler = prepare_dataframe(df_for_training18,columns_to_use_7,0)
trainX19_7, trainY19_7, train_scaler = prepare_dataframe(df_for_training19,columns_to_use_7,0)
trainX20_7, trainY20_7, train_scaler = prepare_dataframe(df_for_training20,columns_to_use_7,0)
trainX21_7, trainY21_7, train_scaler = prepare_dataframe(df_for_training21,columns_to_use_7,0)




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

trainX1_max, trainY1_max, train_scaler = prepare_dataframe(df_for_training1,columns_to_use_max,0)
trainX2_max, trainY2_max, train_scaler = prepare_dataframe(df_for_training2,columns_to_use_max,0)
trainX3_max, trainY3_max, train_scaler = prepare_dataframe(df_for_training3,columns_to_use_max,0)
trainX4_max, trainY4_max, train_scaler = prepare_dataframe(df_for_training4,columns_to_use_max,0)
trainX5_max, trainY5_max, train_scaler = prepare_dataframe(df_for_training5,columns_to_use_max,0)
trainX6_max, trainY6_max, train_scaler = prepare_dataframe(df_for_training6,columns_to_use_max,0)
trainX7_max, trainY7_max, train_scaler = prepare_dataframe(df_for_training7,columns_to_use_max,0)
trainX8_max, trainY8_max, train_scaler = prepare_dataframe(df_for_training8,columns_to_use_max,0)
trainX9_max, trainY9_max, train_scaler = prepare_dataframe(df_for_training9,columns_to_use_max,0)
trainX10_max, trainY10_max, train_scaler = prepare_dataframe(df_for_training10,columns_to_use_max,0)
trainX11_max, trainY11_max, train_scaler = prepare_dataframe(df_for_training11,columns_to_use_max,0)

trainX12_max, trainY12_max, train_scaler = prepare_dataframe(df_for_training12,columns_to_use_max,0)
trainX13_max, trainY13_max, train_scaler = prepare_dataframe(df_for_training13,columns_to_use_max,0)
trainX14_max, trainY14_max, train_scaler = prepare_dataframe(df_for_training14,columns_to_use_max,0)
trainX15_max, trainY15_max, train_scaler = prepare_dataframe(df_for_training15,columns_to_use_max,0)
trainX16_max, trainY16_max, train_scaler = prepare_dataframe(df_for_training16,columns_to_use_max,0)
trainX17_max, trainY17_max, train_scaler = prepare_dataframe(df_for_training17,columns_to_use_max,0)
trainX18_max, trainY18_max, train_scaler = prepare_dataframe(df_for_training18,columns_to_use_max,0)
trainX19_max, trainY19_max, train_scaler = prepare_dataframe(df_for_training19,columns_to_use_max,0)
trainX20_max, trainY20_max, train_scaler = prepare_dataframe(df_for_training20,columns_to_use_max,0)
trainX21_max, trainY21_max, train_scaler = prepare_dataframe(df_for_training21,columns_to_use_max,0)


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




#20 stock basket
#use only Close column
model_1 = build_lstm_model(input_shape=(n_past, len(columns_to_use_1)))
model_1.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mse')
big_train_X_1 = np.concatenate((trainX1_1,trainX2_1,trainX3_1,trainX4_1,trainX5_1,trainX6_1,trainX7_1,trainX8_1,trainX9_1,trainX10_1,trainX11_1,
                                trainX12_1,trainX13_1,trainX14_1,trainX15_1,trainX16_1,trainX17_1,trainX18_1,trainX19_1,trainX20_1,trainX21_1),axis=0)
big_train_Y_1 = np.concatenate((trainY1_1,trainY2_1,trainY3_1,trainY4_1,trainY5_1,trainY6_1,trainY7_1,trainY8_1,trainY9_1,trainY10_1,trainY11_1,
                                trainY12_1,trainY13_1,trainY14_1,trainY15_1,trainY16_1,trainY17_1,trainY18_1,trainY19_1,trainY20_1,trainY21_1),axis=0)
history_1 = model_1.fit(big_train_X_1, big_train_Y_1, epochs=15, batch_size=16, validation_split=0.3, verbose=0)

model_name = "model_1"
model_folder = '/Users/leonbozianu/work/lightbox/models/{}'.format(model_name)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model_1.save_weights(filepath=model_folder+'/final_weight.h5')



#20 stock basket
#use Close, ATR10 columns
print(trainX10_2.shape)
model_2 = build_lstm_model(input_shape=(n_past, len(columns_to_use_2)))
model_2.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mse')
big_train_X_2 = np.concatenate((trainX1_2,trainX2_2,trainX3_2,trainX4_2,trainX5_2,trainX6_2,trainX7_2,trainX8_2,trainX9_2,trainX10_2,trainX11_2,
                                trainX12_2,trainX13_2,trainX14_2,trainX15_2,trainX16_2,trainX17_2,trainX18_2,trainX19_2,trainX20_2,trainX21_2),axis=0)
big_train_Y_2 = np.concatenate((trainY1_2,trainY2_2,trainY3_2,trainY4_2,trainY5_2,trainY6_2,trainY7_2,trainY8_2,trainY9_2,trainY10_2,trainY11_2,
                                trainY12_2,trainY13_2,trainY14_2,trainY15_2,trainY16_2,trainY17_2,trainY18_2,trainY19_2,trainY20_2,trainY21_2),axis=0)
history_2 = model_2.fit(big_train_X_2, big_train_Y_2, epochs=15, batch_size=16, validation_split=0.3, verbose=0)
model_folder = '/Users/leonbozianu/work/lightbox/models/{}'.format("model_2")
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model_2.save_weights(filepath=model_folder+'/final_weight.h5')


#20 stock basket
#use OHLCV columns
model_5 = build_lstm_model(input_shape=(n_past, len(columns_to_use_5)))
model_5.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mse')
big_train_X_5 = np.concatenate((trainX1_5,trainX2_5,trainX3_5,trainX4_5,trainX5_5,trainX6_5,trainX7_5,trainX8_5,trainX9_5,trainX10_5,trainX11_5,
                                trainX12_5,trainX13_5,trainX14_5,trainX15_5,trainX16_5,trainX17_5,trainX18_5,trainX19_5,trainX20_5,trainX21_5),axis=0)
big_train_Y_5 = np.concatenate((trainY1_5,trainY2_5,trainY3_5,trainY4_5,trainY5_5,trainY6_5,trainY7_5,trainY8_5,trainY9_5,trainY10_5,trainY11_5,
                                trainY12_5,trainY13_5,trainY14_5,trainY15_5,trainY16_5,trainY17_5,trainY18_5,trainY19_5,trainY20_5,trainY21_5),axis=0)
history_5 = model_5.fit(big_train_X_5, big_train_Y_5, epochs=15, batch_size=16, validation_split=0.3, verbose=0)
model_folder = '/Users/leonbozianu/work/lightbox/models/{}'.format("model_5")
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model_5.save_weights(filepath=model_folder+'/final_weight.h5')


#20 stock basket
#use 7 indics columns
model_7 = build_lstm_model(input_shape=(n_past, len(columns_to_use_7)))
model_7.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mse')
big_train_X_7 = np.concatenate((trainX1_7,trainX2_7,trainX3_7,trainX4_7,trainX5_7,trainX6_7,trainX7_7,trainX8_7,trainX9_7,trainX10_7,trainX11_7,
                                trainX12_7,trainX13_7,trainX14_7,trainX15_7,trainX16_7,trainX17_7,trainX18_7,trainX19_7,trainX20_7,trainX21_7),axis=0)
big_train_Y_7 = np.concatenate((trainY1_7,trainY2_7,trainY3_7,trainY4_7,trainY5_7,trainY6_7,trainY7_7,trainY8_7,trainY9_7,trainY10_7,trainY11_7,
                                trainY12_7,trainY13_7,trainY14_7,trainY15_7,trainY16_7,trainY17_7,trainY18_7,trainY19_7,trainY20_7,trainY21_7),axis=0)
history_7 = model_7.fit(big_train_X_7, big_train_Y_7, epochs=15, batch_size=16, validation_split=0.3, verbose=0)
model_folder = '/Users/leonbozianu/work/lightbox/models/{}'.format("model_7")
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model_7.save_weights(filepath=model_folder+'/final_weight.h5')


#20 stock basket
#use OHLCV column
model_max = build_lstm_model(input_shape=(n_past, len(columns_to_use_max)))
model_max.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mse')
big_train_X_max = np.concatenate((trainX1_max,trainX2_max,trainX3_max,trainX4_max,trainX5_max,trainX6_max,trainX7_max,trainX8_max,trainX9_max,trainX10_max,trainX11_max,
                                trainX12_max,trainX13_max,trainX14_max,trainX15_max,trainX16_max,trainX17_max,trainX18_max,trainX19_max,trainX20_max,trainX21_max),axis=0)
big_train_Y_max = np.concatenate((trainY1_max,trainY2_max,trainY3_max,trainY4_max,trainY5_max,trainY6_max,trainY7_max,trainY8_max,trainY9_max,trainY10_max,trainY11_max,
                                trainY12_max,trainY13_max,trainY14_max,trainY15_max,trainY16_max,trainY17_max,trainY18_max,trainY19_max,trainY20_max,trainY21_max),axis=0)
history_max = model_max.fit(big_train_X_max, big_train_Y_max, epochs=15, batch_size=16, validation_split=0.3, verbose=0)
model_folder = '/Users/leonbozianu/work/lightbox/models/{}'.format("model_max")
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model_max.save_weights(filepath=model_folder+'/final_weight.h5')



plt.figure(figsize=(4,3))

plt.plot(history_1.history['loss'], label='Train loss Close')
plt.plot(history_1.history['val_loss'], label='Val loss Close')
plt.plot(history_2.history['loss'], label='Train loss 2')
plt.plot(history_2.history['val_loss'], label='Val loss 2')
plt.plot(history_5.history['loss'], label='Train loss OHLCV')
plt.plot(history_5.history['val_loss'], label='Val loss OHLCV')
plt.plot(history_7.history['loss'], label='Train loss 7')
plt.plot(history_7.history['val_loss'], label='Val loss 7')
plt.plot(history_max.history['loss'], label='Train loss Max')
plt.plot(history_max.history['val_loss'], label='Val loss Max')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize='x-small')
plt.show()


