# Ethereum Price Prediction

Author: Utsav Singh
Description:
A machine learning project to predict Ethereum (ETH-USD) prices using Long Short-Term Memory (LSTM) neural networks. This project fetches historical Ethereum price data using `yfinance`, processes it for training, and predicts future prices while visualizing the results.

## Table of Contents

- [Project Overview](#Project-Overview)
- [Key features](#Key-features)
- [Setup and Installation](#Setup-and-Installation)
- [Project Structure](#project-structure)
- [Code Explanation](#Code-Explanation)

## Project Overview
In this project, we use yfinance to gather historical price data for Ethereum and preprocess it for training the LSTM model. The model is then trained on this data and used to predict the next 30 days of Ethereum prices. The predicted prices are visualized to help users understand the trend and potential future movements in the price of Ethereum.

This project demonstrates how machine learning can be applied to cryptocurrency price forecasting, offering valuable insights for investors and enthusiasts looking to make data-driven decisions in the cryptocurrency market.

## Key features

- Data Collection: Fetches historical Ethereum price data using yfinance.
- Data Preprocessing: Scales and reshapes the data to fit the LSTM input format.
- Model Training: Uses LSTM networks to train on historical Ethereum price data.
- Price Prediction: Predicts Ethereum's price for the next 30 days.
- Visualization: Displays actual vs predicted price data using graphs to visualize trends.

## Setup and Installation

To run this project locally, please follow the steps below:
1. Clone the repository or save the code as app.py.
2. Install the required packages by running:
```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow
```

## Project Structure

The project has the following structure:
notebook for training and predictions │ ├── crypto_price_prediction.py # Python script to train and predict cryptocurrency prices ├── requirements.txt # Required libraries and dependencies └── README.md # This file

## Code Explanation

# Importing Libraries
```bash
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta     # For date manipulation
from sklearn.metrics import mean_squared_error  # For evaluating model performance
```
- yfinance: Used to fetch historical price data from Yahoo Finance.
- pandas: For data manipulation and handling.
- numpy: For numerical operations, especially on arrays.
- matplotlib: For visualizing the results (plots).
- MinMaxScaler: To scale the data between 0 and 1, which is required for neural network training.
- tensorflow.keras: Used for building the LSTM (Long Short-Term Memory) model for time series prediction.
- datetime.timedelta: Adds or subtracts a specific duration to/from a given date, essential for generating future dates in predictions.
- sklearn.metrics.mean_squared_error: Evaluates model accuracy by calculating the Mean Squared Error between actual and predicted prices.

# Fetching Historical Ethereum Data
The data for Ethereum (ETH) is fetched using yfinance. You can specify the ticker symbol ('ETH-USD') and the time period you are interested in:
```bash
eth_data = yf.download('ETH-USD', start='2023-01-01', end='2024-12-01')
```
- The start and end parameters define the time range for the data. In this case, it pulls data for Ethereum between January 1, 2023, and January 1, 2024.
- The data DataFrame will contain the historical price data, including columns like Open, High, Low, Close, and Volume.


To print the first few rows of a dataset called eth_data in Python using pandas, the head() function is used
```
print(eth_data.head())
```
 








