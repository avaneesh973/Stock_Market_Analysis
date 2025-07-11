import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def load_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    df['Log_Close'] = np.log(df['Close'])
    return df


def rolling_arima_forecast(df, train_end_date, test_start_date, order=(1, 1, 1), window_size=120):
    train = df[df.index < train_end_date]['Log_Close']
    test = df[df.index >= test_start_date]['Log_Close']

    history = train.copy()
    predictions_log = []

    for t in range(len(test)):
        window = history[-window_size:] if len(history) > window_size else history
        model = ARIMA(window, order=order)
        model_fit = model.fit()
        forecast_log = model_fit.forecast(steps=1)
        predictions_log.append(forecast_log.values[0])
        history = pd.concat([history, pd.Series([test.iloc[t]], index=[test.index[t]])])

    predicted_prices = np.exp(predictions_log)
    actual_prices = np.exp(test.values)

    return actual_prices, predicted_prices, test.index
