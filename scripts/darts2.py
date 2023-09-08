from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

from darts import TimeSeries
from darts.models import NBEATSModel, RNNModel
from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

ticker_symbol = "CL=F"  # This is the ticker symbol for gold futures

# Define the start and end dates for the data
start_date = datetime.datetime.now() - datetime.timedelta(days=15*365)  
end_date = datetime.datetime.now()
data = yf.download(ticker_symbol, start=start_date, end=end_date)
# data.reset_index(inplace=True)
print(data)
print(data.columns)

dfm = data.resample('M').mean()
dfm = dfm.reset_index()

# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(dfm, "Date", "Close")




close_series_train,  close_series_val = series[:-12], series[-24:]
close_series_train.plot()
close_series_val.plot()
plt.title(ticker_symbol)
plt.legend()
plt.show()


# create month and year covariate series
year_series = datetime_attribute_timeseries(
    pd.date_range(start=series.start_time(), freq=series.freq_str, periods=1000),
    attribute="year",
    one_hot=False,
)
year_series = Scaler().fit_transform(year_series)
month_series = datetime_attribute_timeseries(
    year_series, attribute="month", one_hot=True
)
covariates = year_series.stack(month_series).astype(np.float32)
cov_train, cov_val = covariates[:-12], covariates[-24:]


scaler_tr = Scaler()
train_transformed = scaler_tr.fit_transform(close_series_train).astype(np.float32)
val_transformed = scaler_tr.transform(close_series_val).astype(np.float32)

# model = NBEATSModel(input_chunk_length=24 , output_chunk_length=12, n_epochs=100 , random_state=1)
# model.fit([train_transformed],verbose=True)
# model.save("NBEATS_model.pkl")
# pred = model.predict(n=12, series=train_transformed)



my_model = RNNModel(
    model="LSTM",
    hidden_dim=20,
    dropout=0,
    batch_size=16,
    n_epochs=300,
    optimizer_kwargs={"lr": 1e-3},
    model_name="RNN",
    random_state=42,
    training_length=20,
    input_chunk_length=14,
    force_reset=True,
)


my_model.fit(
    train_transformed,
    future_covariates=covariates,
    val_series=val_transformed,
    val_future_covariates=covariates,
    verbose=True,
)
my_model.save("LSTM_model.pkl")


def eval_model(model):
    pred_series = model.predict(n=26, future_covariates=covariates)
    plt.figure(figsize=(8, 5))
    train_transformed.plot(label="Close")
    val_transformed.plot(label="Truth")
    pred_series.plot(label="forecast")
    plt.title("{} MAPE: {:.2f}%".format(ticker_symbol,mape(pred_series, val_transformed)))
    plt.legend()
    plt.ylabel('Scaled Close')
    plt.savefig("/Users/leonbozianu/work/lightbox/compare/"+"darts2a.png")


eval_model(my_model)



series_transformed = scaler_tr.transform(series).astype(np.float32)

backtest_series = my_model.historical_forecasts(
    series_transformed,
    future_covariates=covariates,
    start=pd.Timestamp("20120101"),
    forecast_horizon=6,
    retrain=False,
    verbose=True,
)



plt.figure(figsize=(8, 5))
series_transformed.plot(label="actual")
backtest_series.plot(label="backtest")
plt.legend()
plt.ylabel('Scaled Close')
plt.title(f"{ticker_symbol} Backtest, starting Jan 2012, 6-months horizon")
plt.savefig("/Users/leonbozianu/work/lightbox/compare/"+"darts2.png")

print(
    "MAPE: {:.2f}%".format(
        mape(
            scaler_tr.inverse_transform(series_transformed),
            scaler_tr.inverse_transform(backtest_series),
        )
    )
)