# auto_pipeline_plus.py

import os
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Alpaca-Py
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
USE_PAPER = True  # Paper trading mode

trading_client = TradingClient(API_KEY, API_SECRET, paper=USE_PAPER)

# =========================
# DATA FETCHING
# =========================
def fetch_data(ticker="AAPL", period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    # Clean column names in case of multi-index (yfinance sometimes returns 2D columns)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    return df

# =========================
# FEATURE ENGINEERING
# =========================
def add_features(df):
    # Ensure 'Close' is 1-D
    if isinstance(df["Close"], pd.DataFrame):
        close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]

    # Indicators
    df["RSI"] = RSIIndicator(close).rsi()
    macd = MACD(close)
    df["MACD"] = macd.macd()
    df["Signal"] = macd.macd_signal()

    df.dropna(inplace=True)
    return df

# =========================
# ML MODEL
# =========================
def train_model(df):
    X = df[["RSI", "MACD", "Signal"]]
    y = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("ML Report:")
    print(classification_report(y_test, preds))

    joblib.dump(model, "model.pkl")
    return model

def load_model():
    return joblib.load("model.pkl")

# =========================
# TRADING LOGIC
# =========================
def predict_and_trade(ticker="AAPL"):
    df = fetch_data(ticker)
    df = add_features(df)
    model = load_model()

    X_latest = df[["RSI", "MACD", "Signal"]].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(X_latest)[0]

    if prediction == 1:
        print(f"Model predicts {ticker} will go UP → BUY signal")
        place_order(ticker, qty=1, side=OrderSide.BUY)
    else:
        print(f"Model predicts {ticker} will go DOWN → SELL signal")
        place_order(ticker, qty=1, side=OrderSide.SELL)

# =========================
# ORDER HANDLER
# =========================
def place_order(symbol, qty, side):
    try:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        print(f"✅ Order placed: {side} {qty} {symbol}")
    except Exception as e:
        print("❌ Order failed:", e)

# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":
    ticker = "AAPL"
    df = fetch_data(ticker)
    df = add_features(df)
    train_model(df)
    predict_and_trade(ticker)
