import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np

# Set up Streamlit app
st.set_page_config(page_title="Stock Market Data analysis", layout="wide")
st.title("📈 Stock Market Data analysis")
st.markdown("### A user-friendly dashboard to analyze sector performance, risk, and stock trends in India")


# Fetch live stock data from Yahoo Finance
@st.cache_data
def fetch_stock_list():
    try:
        # Fetch list of stocks dynamically from Wikipedia S&P 500 page
        stock_symbols = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
        return pd.DataFrame({"Ticker": stock_symbols})
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()


stock_list = fetch_stock_list()
if stock_list.empty:
    st.error("⚠️ No stock data available.")
    st.stop()

# Sidebar for user selection
time_range = st.sidebar.selectbox("Select Time Range", ["1y", "5y", "10y", "max"])
selected_stock = st.sidebar.selectbox("Choose a Stock", stock_list["Ticker"].unique())


# Fetch historical data for selected stock
@st.cache_data
def get_stock_data(ticker, period):
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            raise ValueError(f"No historical data found for {ticker}")
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()


stock_data = get_stock_data(selected_stock, time_range)
if stock_data.empty:
    st.warning("⚠️ No data available for the selected stock.")
    st.stop()

# Display stock chart
if not stock_data.empty:
    st.subheader(f"📊 {selected_stock} Stock Performance")
    fig = px.line(stock_data, x=stock_data.index, y='Close', title=f"{selected_stock} Closing Prices")
    st.plotly_chart(fig)

    # Compute Technical Indicators
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['Volatility'] = stock_data['Close'].pct_change().rolling(20).std()

    # Exponential Moving Averages (EMA)
    stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
    stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()

    # Bollinger Bands
    stock_data['Middle Band'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['Upper Band'] = stock_data['Middle Band'] + (stock_data['Close'].rolling(window=20).std() * 2)
    stock_data['Lower Band'] = stock_data['Middle Band'] - (stock_data['Close'].rolling(window=20).std() * 2)

    # Moving Average Convergence Divergence (MACD)
    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
    stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    # Relative Strength Index (RSI)
    avg_gain = stock_data['Close'].diff().where(stock_data['Close'].diff() > 0, 0).rolling(14).mean().fillna(0)
    avg_loss = -stock_data['Close'].diff().where(stock_data['Close'].diff() < 0, 0).rolling(14).mean().fillna(0)
    avg_loss = avg_loss.replace(0, 1e-6)
    rs = avg_gain / avg_loss.fillna(1)
    stock_data['RSI'] = 100 - (100 / (1 + rs))

# AI-Based Stock Recommendations
st.subheader("📊 AI-Based Stock Recommendation")


def stock_recommendation(stock_data):
    try:
        if stock_data.empty:
            return "⚠️ No Data Available"
        latest_rsi = stock_data['RSI'].iloc[-1]
        latest_macd = stock_data['MACD'].iloc[-1]
        latest_signal = stock_data['Signal Line'].iloc[-1]
        latest_volatility = stock_data['Volatility'].iloc[-1]
        latest_sma_50 = stock_data['SMA_50'].iloc[-1]
        latest_sma_200 = stock_data['SMA_200'].iloc[-1]

        previous_sma_50 = stock_data['SMA_50'].iloc[-2]
        previous_sma_200 = stock_data['SMA_200'].iloc[-2]
        golden_cross = previous_sma_50 < previous_sma_200 and latest_sma_50 > latest_sma_200
        death_cross = previous_sma_50 > previous_sma_200 and latest_sma_50 < latest_sma_200
        macd_histogram = latest_macd - latest_signal

        if latest_rsi > 70 and latest_macd < latest_signal and death_cross:
            return "📉 Strong Sell - Overbought, Bearish MACD & Death Cross"
        elif latest_rsi < 30 and latest_macd > latest_signal and golden_cross:
            return "📈 Strong Buy - Oversold, Bullish MACD & Golden Cross"
        elif macd_histogram > 0 and latest_sma_50 > latest_sma_200:
            return "📈 Buy - Bullish MACD & Uptrend"
        elif macd_histogram < 0 and latest_sma_50 < latest_sma_200:
            return "📉 Sell - Bearish MACD & Downtrend"
        elif latest_volatility > stock_data['Volatility'].quantile(0.75):
            return "⚠️ Hold - High Volatility Detected"
        else:
            return "🔄 Hold - Weak or No Strong Signal"
    except Exception as e:
        return f"Error in recommendation: {e}"


if not stock_data.empty:
    recommendation = stock_recommendation(stock_data)
    st.markdown(
        f"<div style='padding:10px; border-radius:5px; background-color:#2e86c1; color:white; font-size:18px; font-weight:bold;'>{recommendation}</div>",
        unsafe_allow_html=True)
