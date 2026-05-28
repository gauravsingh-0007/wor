import yfinance as yf
import pandas as pd
import numpy as np
import ta
import streamlit as st
import plotly.graph_objects as go

from textblob import TextBlob
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="AI Stock Predictor",
    layout="wide"
)

st.title("📈 AI Stock Market Prediction System")

stock = st.text_input(
    "Enter Stock Symbol",
    "AAPL"
).upper()


@st.cache_data
def load_data(symbol):
    return yf.download(
        symbol,
        start="2020-01-01",
        end="2025-01-01",
        auto_adjust=True
    )


try:
    data = load_data(stock)

except Exception as e:
    st.error(f"Error Fetching Data: {e}")
    st.stop()

if data.empty:
    st.error("No stock data found")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

st.subheader("📊 Stock Dataset")

st.write(data.tail())

close = pd.Series(data['Close']).squeeze()

data['MA20'] = close.rolling(window=20).mean()

data['RSI'] = ta.momentum.RSIIndicator(close).rsi()

data['MACD'] = ta.trend.MACD(close).macd()

data['Volume_Avg'] = data['Volume'].rolling(window=20).mean()

data.dropna(inplace=True)

st.subheader("🕯️ Candlestick Chart")

fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)])

fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📉 Closing Price & MA20")

chart_data = pd.DataFrame({
    'Close Price': data['Close'],
    'MA20': data['MA20']
})

st.line_chart(chart_data)

close_data = data[['Close']].copy()

close_data['Prediction'] = close_data['Close'].shift(-1)

dataset = close_data.dropna()

X = np.array(dataset[['Close']])

Y = np.array(dataset['Prediction'])

model = LinearRegression()

model.fit(X, Y)

last_close = np.array([[data['Close'].iloc[-1]]])

pred_price = model.predict(last_close)

st.subheader("💰 Predicted Next Day Price")

st.success(f"Predicted Price: ${pred_price[0]:.2f}")

st.subheader("📰 News Sentiment Analysis")

sample_news = [
    "Company profits increased significantly",
    "Strong quarterly earnings reported",
    "Investors are optimistic about growth"
]

positive_score = 0

for news in sample_news:

    sentiment = TextBlob(news).sentiment.polarity

    if sentiment > 0:
        positive_score += 1

    st.write(f"News: {news}")
    st.write(f"Sentiment Score: {sentiment:.2f}")

st.subheader("🧠 Why Stock May Increase?")

reasons = []

latest_rsi = data['RSI'].iloc[-1]

latest_macd = data['MACD'].iloc[-1]

latest_volume = data['Volume'].iloc[-1]

avg_volume = data['Volume_Avg'].iloc[-1]

latest_close = data['Close'].iloc[-1]

latest_ma20 = data['MA20'].iloc[-1]

if latest_rsi > 70:
    reasons.append("RSI shows strong bullish momentum")

elif latest_rsi < 30:
    reasons.append("Stock is oversold and may recover")

if latest_macd > 0:
    reasons.append("MACD indicates bullish trend")

if latest_volume > avg_volume:
    reasons.append("High trading volume detected")

if latest_close > latest_ma20:
    reasons.append("Price is above Moving Average")

if positive_score >= 2:
    reasons.append("Positive market news sentiment")

if not reasons:
    st.info("No strong bullish or bearish indicators detected")

for reason in reasons:
    st.success(reason)

st.subheader("📍 Trading Signal")

if latest_rsi < 30 and latest_macd > 0:
    st.success("BUY SIGNAL")

elif latest_rsi > 70:
    st.error("SELL SIGNAL")

else:
    st.warning("HOLD")

st.subheader("📈 Technical Indicator Values")

st.write(f"RSI: {latest_rsi:.2f}")
st.write(f"MACD: {latest_macd:.2f}")
st.write(f"Latest Close Price: ${latest_close:.2f}")
st.write(f"20-Day Moving Average: ${latest_ma20:.2f}")

st.subheader("🔄 Project Workflow")

st.code("""

User Input Stock
        |
        V
Fetch Historical Data
        |
        V
Calculate Indicators
(RSI, MACD, MA)
        |
        V
News Sentiment Analysis
        |
        V
Train ML Model
        |
        V
Predict Future Price
        |
        V
Generate Explanation
        |
        V
Display Dashboard

""")

st.write("AI Powered Stock Prediction System")