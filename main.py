import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

#downloading the data
df = yf.download("AAPL", start='2010-01-01', end='2025-06-30', auto_adjust=True)

#plotting the close price of the selected company
plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.title('Apple Stock Price (Close)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.savefig('apple.png')

#Calculating moving averages of close price
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
#plotting the close price and moving averages to visualize an approximate trend
plt.figure(figsize=(12,8))
plt.plot(df['Close'])
plt.plot(ma100)
plt.plot(ma200)
plt.savefig('apple_ma.png')

# Removing the COVID Period (Optional)
covid_start = '2020-02-01'
covid_end = '2020-12-31'
data = data[(data.index < covid_start) | (data.index > covid_end)]

#Splitting data into train and test data
train_data = data[(data.index < '2025-05-31')]
test_data = data[(data.index > '2025-05-31')]

# Calculate daily log returns
train_data['Log_Returns'] = np.log(train_data['Close'] / train_data['Close'].shift(1))
train_data.dropna(inplace=True)

# Fit ARIMA on log returns (or differenced data)
model = ARIMA(train_data['Log_Returns'], order=(1,0,1))
model_fit = model.fit()

#Forecasting June 2025
