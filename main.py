import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Load the stock data
df = yf.download("AAPL", start='2015-01-01', end='2025-06-30', auto_adjust=True)

df['Log_Close'] = np.log(df['Close'])

# Step 2: Split data into training and testing (June 2025 = test set)
train = df[df.index < '2025-04-01']['Log_Close']
test = df[(df.index >= '2025-04-01')]['Log_Close']

print(f"Train length: {len(train)}")
print(f"Test length: {len(test)}")

# Step 3: Rolling Forecast with fixed (p,d,q) and sliding window (last 120 days)
history = train.copy()
predictions_log = []

for t in range(len(test)):
    window = history[-120:] if len(history) > 120 else history
    
    model = ARIMA(window, order=(1,1,1))  # d=1 handles differencing
    model_fit = model.fit()
    
    forecast_log = model_fit.forecast(steps=1)
    predictions_log.append(forecast_log)
    
    history = pd.concat([history, pd.Series([test.iloc[t]], index=[test.index[t]])])

# Convert both predictions and actual test data back to normal prices
predicted_prices = np.exp(predictions_log)
actual_prices = np.exp(test.values)

plt.figure(figsize=(12,6))
#plt.plot(df[df.index < '2025-06-01'].index, df[df.index < '2025-06-01']['Close'], label='Train (Original Prices)')
plt.plot(df[(df.index >= '2025-04-01')].index, actual_prices, label='Actual Prices')
plt.plot(df[(df.index >= '2025-04-01')].index, predicted_prices, label='Forecast Prices', linestyle='--')
plt.legend()
plt.title('Rolling ARIMA Forecast (on Log Prices, Converted Back)')
plt.savefig('new.png')

from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mae = mean_absolute_error(actual_prices, predicted_prices)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')
