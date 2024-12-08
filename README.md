# Ethereum Price Prediction

Author: Utsav Singh
Description:
This project leverages machine learning, specifically Long Short-Term Memory (LSTM) networks, to predict the future prices of Ethereum (ETH). By analyzing historical price data, the model identifies patterns and trends to make accurate predictions for the next 30 days. The project uses time series forecasting techniques to model the price movements and visualize the predicted values.

## Table of Contents

- [Project Overview](#ProjectOverview)
- [Key features](#Keyfeatures)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

# Project Overview
In this project, we use yfinance to gather historical price data for Ethereum and preprocess it for training the LSTM model. The model is then trained on this data and used to predict the next 30 days of Ethereum prices. The predicted prices are visualized to help users understand the trend and potential future movements in the price of Ethereum.

This project demonstrates how machine learning can be applied to cryptocurrency price forecasting, offering valuable insights for investors and enthusiasts looking to make data-driven decisions in the cryptocurrency market.

#Key features
<ul>
  <li>Data Collection: Fetches historical Ethereum price data using yfinance.</li>
  <li>Data Preprocessing: Scales and reshapes the data to fit the LSTM input format.</li>
  <li>Model Training: Uses LSTM networks to train on historical Ethereum price data.</li>
  <li>Price Prediction: Predicts Ethereum's price for the next 30 days.</li>
  <li>Visualization: Displays actual vs predicted price data using graphs to visualize trends.</li>
</ul>
