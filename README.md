# Financial time series analysis at Lightbox
## Project description
The code in this repository is the result of our secondment work at Lightbox Technologies SA.

### Secondment summary
Lightbox Technologies SA provides consultancy, research and development of hardware and software infrastructures, algorithms, mathematical and statistical models, aimed at the acquisition, maintenance, processing and analysis of massive volumes of data, and the optimization of decision-making and production processes of third-party entities.
The work contained in this repository focuses on the application of machine learning (ML) tools to financial time-series data through the development of systems for predictive analysis using historical market data time series for a multitude of financial instruments. 

Algorithmic biases were prevented by dividing the time series in training and testing periods. A Transfer Learning technique was employed in order to improve the training of the model by combining data from different securities to compensate for the limited availability of historical data. The training sample was identified by selecting datasets showing similar statistical features. The various models provide future predictions in unseen time-series data along with associated uncertainties.

This repository contains the codebase for a number of ML models ranging from the simple to the complex. We utilise and show the training and inference from classical boosted decision trees to Bayesian and recurrent neural networks. Each network architecture informed subsequent designs.
An important part of the problem was the model adapting to changing and often turbulent time series data. In comparison to other ML applications the models are susceptible to going “out-of-date” quickly - thus presenting challenges to out choice of training/validation data.

This repository contains part of the results produced during this project.

### Repository overview


## Code Usage

In order to start working with the code contained in this repo use the `install.sh` (first time) and `setup.sh` scripts found in the `setup` directory. This will create a conda/miniconda environment with the necessary python libraries installed to reproduce our base results.

The code contained in the `src` directory can be used to train and save particular models before inferring on unseen test data using the pre-trained model weights. A simplified workflow is described below.

In the `lstm-scripts` directory the training can be initiated by giving several command line arguments to the training script for example
```
python lstm-train.py --name AAPL --end_date 2014-01-01 --seq_length 10 --epochs 150
```
where the required arguments are `--name` the name of the underlying stock or asset following yahoo finance naming, `--end_date` the final date that will be included in your overall dataset, `--seq_length` the sequence length that the LSTM model considers and `--epochs` the number of epochs to train.
Similarly the inference script can be run with the following example:
```
python lstm-infer.py --name AAPL --end_date 2014-01-01 --seq_length 10
```
with the same command line arguments, excluding `--epochs`. The name of the saved model weights is consistent across `train.py` and `infer.py`.


