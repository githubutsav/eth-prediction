# Ethereum Price Prediction

Author: Utsav Singh
Description:
A machine learning project to predict Ethereum (ETH-USD) prices using Long Short-Term Memory (LSTM) neural networks. This project fetches historical Ethereum price data using `yfinance`, processes it for training, and predicts future prices while visualizing the results.

## Table of Contents

- [Project Overview](#Project-Overview)
- [Key features](#Key-features)
- [Setup and Installation](#Setup-and-Installation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

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

## 






