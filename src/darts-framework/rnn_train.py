import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
import darts.utils.timeseries_generation as tg
from darts.datasets import AirPassengersDataset, EnergyDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.missing_values import fill_missing_values
from darts.utils.likelihood_models import GaussianLikelihood

import warnings

warnings.filterwarnings("ignore")


length = 1000
trend = tg.linear_timeseries(length=length, end_value=4)
season1 = tg.sine_timeseries(length=length, value_frequency=0.1, value_amplitude=1.0)
noise = tg.gaussian_timeseries(length=length, std=0.9)
noise_modulator = (
    tg.sine_timeseries(length=length, value_frequency=0.05)
    + tg.constant_timeseries(length=length, value=1)
) / 2
noise = noise * noise_modulator

target_series = sum([noise, trend, season1]).astype(np.float32)
covariates = noise_modulator.astype(np.float32)
target_train, target_val = target_series.split_after(0.65)

target_train.plot()
target_val.plot()
plt.show()

my_model = RNNModel(
    model="LSTM",
    hidden_dim=20,
    dropout=0,
    batch_size=16,
    n_epochs=50,
    optimizer_kwargs={"lr": 1e-3},
    random_state=0,
    training_length=50,
    input_chunk_length=20,
    likelihood=GaussianLikelihood(),
)

my_model.fit(target_train, future_covariates=covariates, verbose=True)
my_model.save("my_model.pkl")

pred = my_model.predict(80, num_samples=50)
target_val.slice_intersect(pred).plot(label="target")
pred.plot(label="prediction")
plt.show()








