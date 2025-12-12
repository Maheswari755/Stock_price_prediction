import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import mplfinance as mpf

st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<link rel="manifest" href="/static/manifest.json">
""", unsafe_allow_html=True)

st.title("📈 StockVision AI - Stock Analysis & Prediction")

stocks_list = [
    "Choose Option",
    "GOOG", "AAPL", "MSFT", "TSLA", "AMZN", "META",
    "NVDA", "IBM", "NFLX", "INTC", "ORCL", "ADBE"
]

stock = st.selectbox("Select Stock Symbol:", stocks_list)
start = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end = st.date_input("End Date", pd.to_datetime("2025-12-31"))

if st.button("Fetch & Analyze"):

    if stock == "Choose Option":
        st.error("Select a stock first.")
    else:
        with st.spinner("Downloading stock data…"):
            data = yf.download(stock, start=start, end=end)

        if data.empty:
            st.error("No data found for this stock.")
        else:
            data.columns = [c[0] for c in data.columns]

            st.subheader("Dataset Preview")
            st.dataframe(data.head())

            st.subheader("Closing Price Chart")
            fig, ax = plt.subplots(figsize=(12, 5))
            data["Close"].plot(ax=ax)
            plt.title(f"{stock} Closing Price")
            st.pyplot(fig)

            st.subheader("Candlestick Chart")
            candle = mpf.plot(
                data,
                type='candle',
                volume=True,
                style='charles',
                figsize=(12, 6),
                returnfig=True
            )
            st.pyplot(candle[0])

            st.subheader("Technical Indicators")
            data["MA20"] = data["Close"].rolling(20).mean()
            data["MA50"] = data["Close"].rolling(50).mean()
            data["STD20"] = data["Close"].rolling(20).std()
            data["Upper"] = data["MA20"] + 2 * data["STD20"]
            data["Lower"] = data["MA20"] - 2 * data["STD20"]

            delta = data["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            data["RSI"] = 100 - (100 / (1 + rs))

            fig2, ax2 = plt.subplots(figsize=(12, 5))
            data[["Close", "MA20", "MA50", "Upper", "Lower"]].plot(ax=ax2)
            plt.title(f"{stock} Moving Averages & Bollinger Bands")
            st.pyplot(fig2)

            fig3, ax3 = plt.subplots(figsize=(12, 4))
            data["RSI"].plot(ax=ax3)
            plt.title(f"{stock} RSI")
            st.pyplot(fig3)

            st.subheader("LSTM Stock Price Prediction")
            df = data[["Close"]].dropna().values
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df)

            X, y = [], []
            for i in range(60, len(scaled)):
                X.append(scaled[i-60:i])
                y.append(scaled[i])
            X, y = np.array(X), np.array(y)

            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(60, 1)),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")

            with st.spinner("Training model…"):
                model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)

            pred = model.predict(X_test)
            pred = scaler.inverse_transform(pred)
            actual = scaler.inverse_transform(y_test)

            fig4, ax4 = plt.subplots(figsize=(12, 5))
            ax4.plot(actual, label="Actual")
            ax4.plot(pred, label="Predicted")
            plt.title(f"{stock} LSTM Prediction")
            plt.legend()
            st.pyplot(fig4)

            st.success(f"{stock} Analysis Complete!")
