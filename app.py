# app.py

import streamlit as st
import pandas as pd
from auto_pipeline_plus import (
    add_features,
    fetch_data,
    predict_and_trade,
    train_model,
    load_model,
)
from alpaca.trading.client import TradingClient
import os

st.set_page_config(page_title="AI Trading Pipeline", layout="wide")
st.title("📈 AI Trading Pipeline Dashboard")

# =========================
# Alpaca client
# =========================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("Settings")

# Enter up to 3 tickers manually
tickers_input = st.sidebar.text_input(
    "Enter up to 3 tickers separated by commas", "AAPL,MSFT,TSLA"
)
tickers = [t.strip().upper() for t in tickers_input.split(",")][:3]

# Data period / interval
period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y"], index=2)
interval = st.sidebar.selectbox("Data Interval", ["1d", "1h", "30m"], index=0)

# Action
action = st.sidebar.radio("Action", ["Train Model", "Predict & Trade"])

# =========================
# Top 10 tickers by volume (free Yahoo Finance)
# =========================
st.sidebar.subheader("💎 Top 10 Tickers (by volume)")
top10 = ["AAPL","MSFT","TSLA","NVDA","AMZN","GOOGL","META","SPY","QQQ","JPM"]
st.sidebar.write(top10)

# =========================
# Watchlist
# =========================
st.sidebar.subheader("📋 Watchlist")
watchlist = st.sidebar.multiselect("Select tickers to watch", top10, default=tickers)

# =========================
# Main display for each ticker
# =========================
for ticker in watchlist:
    st.header(f"Ticker: {ticker}")

    # Fetch + clean data
    df = fetch_data(ticker, period=period, interval=interval)
    df = add_features(df)

    st.subheader("📊 Data Preview")
    st.dataframe(df.tail())

    st.subheader("📈 Indicators")
    st.line_chart(df[["RSI", "MACD", "Signal"]])  # <- fixed line

    # Action buttons for each ticker
    if action == "Train Model":
        train_model(df)
        st.success(f"✅ Model trained and saved for {ticker}")

    elif action == "Predict & Trade":
        try:
            load_model()
            predict_and_trade(ticker)
            st.success(f"✅ Prediction + order executed for {ticker}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# =========================
# Last 5 Alpaca Paper Trades
# =========================
st.subheader("📝 Last 5 Orders (Paper Trading)")
try:
    orders = trading_client.get_orders(limit=5, status="all", nested=True)
    orders_list = []
    for o in orders:
        orders_list.append(
            {
                "Symbol": o.symbol,
                "Side": o.side,
                "Qty": o.qty,
                "Status": o.status,
                "Filled Avg Price": o.filled_avg_price,
            }
        )
    orders_df = pd.DataFrame(orders_list)
    st.dataframe(orders_df)
except Exception as e:
    st.warning("No orders yet or failed to fetch orders")
