import numpy as np
from matplotlib import pyplot as plt

from darts.datasets import AirPassengersDataset
from darts import TimeSeries
from darts.models import TCNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import QuantileRegression

series = AirPassengersDataset().load()
train, val = series[:-36], series[-36:]

scaler = Scaler()
train = scaler.fit_transform(train).astype(np.float32)
val = scaler.transform(val).astype(np.float32)
series = scaler.transform(series).astype(np.float32)

model = TCNModel(input_chunk_length=30,
                 output_chunk_length=12,
                 dropout=0.1)
model.fit(train, epochs=400)
pred = model.predict(n=36, mc_dropout=True, num_samples=500)

series.plot()
pred.plot(label='forecast')
plt.xlabel('Month')
plt.ylabel('Number Air Passenger')
plt.tight_layout()
plt.savefig('/Users/leonbozianu/work/lightbox/compare/'+'darts1.png')