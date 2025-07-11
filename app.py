import streamlit as st
import matplotlib.pyplot as plt
from model import load_stock_data, rolling_arima_forecast
from chatbot import get_ticker
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

st.set_page_config(layout="wide")

st.title("📈 Stock Price Forecasting with ARIMA")

col1, col2 = st.columns([4, 1])

with col1:
    st.header("Forecast")

    ticker = st.text_input("Enter Stock Ticker (e.g., TSLA)", value='TSLA')

    start_date = '2015-01-01'
    end_date = '2025-06-30'
    train_end_date = '2025-01-01'
    test_start_date = '2025-06-01'

    if st.button("Run Forecast"):
        df = load_stock_data(ticker, start_date, end_date)

        if df is None:
            st.error("❌ No company found with that ticker. Please check the ticker symbol and try again.")
        else:
            actual_prices, predicted_prices, test_index = rolling_arima_forecast(df, train_end_date, test_start_date)

            plt.figure(figsize=(12, 6))
            plt.plot(test_index, actual_prices, label='Actual Prices')
            plt.plot(test_index, predicted_prices, label='Forecast Prices', linestyle='--')
            plt.legend()
            plt.title(f'{ticker} Rolling ARIMA Forecast')
            st.pyplot(plt)

            rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
            mae = mean_absolute_error(actual_prices, predicted_prices)
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

            st.write(f'**RMSE:** {rmse:.2f}')
            st.write(f'**MAE:** {mae:.2f}')
            st.write(f'**MAPE:** {mape:.2f}%')

with col2:
    with st.expander("💬 Chatbot"):
        if 'chat_step' not in st.session_state:
            st.session_state.chat_step = 0

        user_input = st.text_input("Chat here:", key='chat_input')

        if user_input:
            if st.session_state.chat_step == 0:
                if user_input.strip().lower() in ['yes', 'y']:
                    st.write("Type the company name")
                    st.session_state.chat_step = 1
                else:
                    st.write("Okay! Let me know if you need any help.")
            elif st.session_state.chat_step == 1:
                company_name = user_input.strip()
                ticker_result = get_ticker(company_name)
                if ticker_result:
                    st.write(f"The ticker for {company_name} is **{ticker_result}**")
                else:
                    st.write("Company not found. Please check the name or spelling.")
                st.session_state.chat_step = 0
