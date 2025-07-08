import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

df = yf.download("AAPL", start='2010-01-01', end='2025-06-30', auto_adjust=True)

plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.title('Apple Stock Price (Close)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.savefig('apple.png')

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

plt.figure(figsize=(12,8))
plt.plot(df['Close'])
plt.plot(ma100)
plt.plot(ma200)
plt.savefig('apple_ma.png')