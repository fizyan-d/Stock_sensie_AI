import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np
import time
from fpdf import FPDF
import tempfile, os
from datetime import datetime

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="StockSense AI", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp {
    background: #060912;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99,102,241,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(16,185,129,0.06) 0%, transparent 60%);
}
[data-testid="stSidebar"] {
    background: #0c1018 !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
.header-banner {
    background: linear-gradient(135deg, #0f1623 0%, #131d2e 50%, #0f1623 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px; padding: 28px 36px; margin-bottom: 24px;
    position: relative; overflow: hidden;
}
.header-banner::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #10b981, transparent);
}
.header-title {
    font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 600;
    background: linear-gradient(135deg, #a5b4fc, #6ee7b7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}
.header-sub { color: #475569; font-size: 13px; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
.metric-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 24px; }
.metric-card {
    background: linear-gradient(135deg, #0f1623, #141e30);
    border: 1px solid rgba(99,102,241,0.15); border-radius: 14px;
    padding: 18px 16px; text-align: center; position: relative; overflow: hidden;
}
.metric-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #6366f1, #10b981); opacity: 0.4;
}
.metric-label { color: #475569; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
.metric-value { color: #e2e8f0; font-size: 20px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-change-up   { color: #10b981; font-size: 12px; margin-top: 3px; }
.metric-change-down { color: #ef4444; font-size: 12px; margin-top: 3px; }
.verdict-wrapper { background: linear-gradient(135deg, #0f1623, #141e30); border-radius: 20px; padding: 32px; text-align: center; position: relative; overflow: hidden; }
.verdict-buy  { border: 1px solid rgba(16,185,129,0.4);  box-shadow: 0 0 40px rgba(16,185,129,0.1); }
.verdict-sell { border: 1px solid rgba(239,68,68,0.4);   box-shadow: 0 0 40px rgba(239,68,68,0.1); }
.verdict-hold { border: 1px solid rgba(245,158,11,0.4);  box-shadow: 0 0 40px rgba(245,158,11,0.1); }
.verdict-badge { display: inline-block; padding: 6px 16px; border-radius: 100px; font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 12px; }
.badge-buy  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-sell { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-hold { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.verdict-text-buy  { font-size: 42px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #10b981; letter-spacing: -1px; }
.verdict-text-sell { font-size: 42px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #ef4444; letter-spacing: -1px; }
.verdict-text-hold { font-size: 42px; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #f59e0b; letter-spacing: -1px; }
.conf-bar-bg { background: #1e293b; border-radius: 100px; height: 6px; margin: 14px 0 6px; }
.conf-bar-fill-buy  { height: 6px; border-radius: 100px; background: linear-gradient(90deg, #10b981, #34d399); }
.conf-bar-fill-sell { height: 6px; border-radius: 100px; background: linear-gradient(90deg, #ef4444, #f87171); }
.conf-bar-fill-hold { height: 6px; border-radius: 100px; background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.signal-row { display: flex; align-items: flex-start; gap: 14px; background: #0c1018; border: 1px solid #1e293b; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }
.signal-dot { width: 10px; height: 10px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }
.dot-buy  { background: #10b981; box-shadow: 0 0 8px rgba(16,185,129,0.6); }
.dot-sell { background: #ef4444; box-shadow: 0 0 8px rgba(239,68,68,0.6); }
.dot-hold { background: #f59e0b; box-shadow: 0 0 8px rgba(245,158,11,0.6); }
.signal-name { color: #94a3b8; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.signal-verdict-buy  { color: #10b981; font-size: 13px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.signal-verdict-sell { color: #ef4444; font-size: 13px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.signal-verdict-hold { color: #f59e0b; font-size: 13px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.signal-reason { color: #475569; font-size: 12px; margin-top: 2px; }
.section-header { display: flex; align-items: center; gap: 10px; margin: 28px 0 16px; }
.section-line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(99,102,241,0.3), transparent); }
.section-title { color: #64748b; font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; white-space: nowrap; }
.sidebar-logo { font-family: 'JetBrains Mono', monospace; font-size: 18px; font-weight: 700; background: linear-gradient(135deg, #a5b4fc, #6ee7b7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sidebar-tag { font-size: 11px; color: #334155; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }
div[data-testid="metric-container"] { background: #0f1623; border: 1px solid #1e293b; border-radius: 10px; padding: 12px; }
.stButton > button { background: #0f1623 !important; border: 1px solid #1e293b !important; color: #94a3b8 !important; border-radius: 8px !important; font-size: 12px !important; font-family: 'Space Grotesk', sans-serif !important; transition: all 0.2s !important; }
.stButton > button:hover { border-color: #6366f1 !important; color: #a5b4fc !important; }

/* Portfolio styles */
.portfolio-row {
    display: grid; grid-template-columns: 1.2fr 1fr 1fr 1fr 1fr 1.2fr;
    gap: 8px; align-items: center;
    background: #0c1018; border: 1px solid #1e293b;
    border-radius: 12px; padding: 14px 18px; margin-bottom: 8px;
}
.portfolio-header {
    display: grid; grid-template-columns: 1.2fr 1fr 1fr 1fr 1fr 1.2fr;
    gap: 8px; padding: 6px 18px; margin-bottom: 4px;
}
.port-label { color: #334155; font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.port-sym { color: #a5b4fc; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 14px; }
.port-val { color: #e2e8f0; font-family: 'JetBrains Mono', monospace; font-size: 13px; }
.port-profit { color: #10b981; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; }
.port-loss   { color: #ef4444; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; }
.port-summary {
    background: linear-gradient(135deg, #0f1623, #131d2e);
    border: 1px solid rgba(99,102,241,0.2); border-radius: 16px;
    padding: 20px 24px; margin-bottom: 20px;
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; text-align: center;
}
.pred-card {
    background: linear-gradient(135deg, #0f1623, #141e30);
    border: 1px solid rgba(99,102,241,0.2); border-radius: 14px;
    padding: 16px; text-align: center;
}
.vol-gauge-wrap {
    background: linear-gradient(135deg, #0f1623, #141e30);
    border: 1px solid #1e293b; border-radius: 16px; padding: 24px; text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
INDIAN_STOCKS = {
    "TCS","INFY","RELIANCE","HDFCBANK","ICICIBANK","WIPRO","SBIN",
    "AXISBANK","BAJFINANCE","HINDUNILVR","ITC","LT","KOTAKBANK",
    "MARUTI","NESTLEIND","ONGC","POWERGRID","SUNPHARMA","TATAMOTORS",
    "TATASTEEL","TECHM","TITAN","ULTRACEMCO","ADANIENT","ADANIPORTS",
    "APOLLOHOSP","ASIANPAINT","BAJAJFINSV","BPCL","BRITANNIA",
    "CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM",
    "HCLTECH","HEROMOTOCO","HINDALCO","JSWSTEEL","NTPC","UPL","ZOMATO",
    "NYKAA","PAYTM","IRCTC","HAL","BEL","DMART","PIDILITIND"
}
PLOT_BG = PAPER_BG = '#060912'
GRID_COLOR = '#0f1623'
FONT_COLOR = '#64748b'


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_currency(symbol):
    if symbol.endswith('.NS') or symbol.endswith('.BO'):
        return '₹', 'INR'
    return '$', 'USD'

def fmt_price(val, symbol):
    cur, _ = get_currency(symbol)
    return f"{cur}{val:,.2f}"

@st.cache_data(ttl=300)
def fetch_stock_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        info = ticker.info
        return df, info
    except Exception:
        return None, None

@st.cache_data(ttl=600)
def fetch_current_price(symbol):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="2d", interval="1d")
        if df is not None and not df.empty:
            return float(df['Close'].iloc[-1])
    except Exception:
        pass
    return None

def calculate_indicators(df):
    df = df.copy()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI']  = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        cols = macd.columns.tolist()
        mc = [c for c in cols if c.startswith('MACD_')]
        sc = [c for c in cols if c.startswith('MACDs_')]
        hc = [c for c in cols if c.startswith('MACDh_')]
        df['MACD']        = macd[mc[0]] if mc else 0
        df['MACD_Signal'] = macd[sc[0]] if sc else 0
        df['MACD_Hist']   = macd[hc[0]] if hc else 0
    else:
        df['MACD'] = df['MACD_Signal'] = df['MACD_Hist'] = 0
    bb = ta.bbands(df['Close'], length=20, std=2)
    if bb is not None:
        bc = bb.columns.tolist()
        df['BB_Upper'] = bb[[c for c in bc if c.startswith('BBU')][0]]
        df['BB_Lower'] = bb[[c for c in bc if c.startswith('BBL')][0]]
        df['BB_Mid']   = bb[[c for c in bc if c.startswith('BBM')][0]]
    else:
        df['BB_Upper'] = df['BB_Lower'] = df['BB_Mid'] = df['Close']
    return df

def generate_recommendation(df):
    signals, reasons = {}, {}
    latest, prev = df.iloc[-1], df.iloc[-2]
    if latest['MA20'] > latest['MA50'] and prev['MA20'] <= prev['MA50']:
        signals['MA Crossover'] = 'BUY';  reasons['MA Crossover'] = 'Golden Cross — short MA just crossed above long MA'
    elif latest['MA20'] < latest['MA50'] and prev['MA20'] >= prev['MA50']:
        signals['MA Crossover'] = 'SELL'; reasons['MA Crossover'] = 'Death Cross — short MA just crossed below long MA'
    elif latest['MA20'] > latest['MA50']:
        signals['MA Crossover'] = 'BUY';  reasons['MA Crossover'] = 'Short MA is above Long MA — bullish trend'
    else:
        signals['MA Crossover'] = 'SELL'; reasons['MA Crossover'] = 'Short MA is below Long MA — bearish trend'
    rsi = latest['RSI']
    if rsi < 30:   signals['RSI'] = 'BUY';  reasons['RSI'] = f'RSI {rsi:.1f} — oversold (below 30)'
    elif rsi > 70: signals['RSI'] = 'SELL'; reasons['RSI'] = f'RSI {rsi:.1f} — overbought (above 70)'
    else:          signals['RSI'] = 'HOLD'; reasons['RSI'] = f'RSI {rsi:.1f} — neutral zone (30–70)'
    if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
        signals['MACD'] = 'BUY';  reasons['MACD'] = 'MACD line just crossed above signal line'
    elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
        signals['MACD'] = 'SELL'; reasons['MACD'] = 'MACD line just crossed below signal line'
    elif latest['MACD'] > latest['MACD_Signal']:
        signals['MACD'] = 'BUY';  reasons['MACD'] = 'MACD above signal line — positive momentum'
    else:
        signals['MACD'] = 'SELL'; reasons['MACD'] = 'MACD below signal line — negative momentum'
    p = latest['Close']
    if p <= latest['BB_Lower']:   signals['Bollinger'] = 'BUY';  reasons['Bollinger'] = 'Price at lower band — likely to bounce up'
    elif p >= latest['BB_Upper']: signals['Bollinger'] = 'SELL'; reasons['Bollinger'] = 'Price at upper band — likely to pull back'
    else:                         signals['Bollinger'] = 'HOLD'; reasons['Bollinger'] = 'Price within bands — no extreme signal'
    buy_c  = sum(1 for s in signals.values() if s == 'BUY')
    sell_c = sum(1 for s in signals.values() if s == 'SELL')
    hold_c = sum(1 for s in signals.values() if s == 'HOLD')
    total  = len(signals)
    if buy_c >= 3:                      verdict, conf = 'STRONG BUY',  int(buy_c/total*100)
    elif buy_c == 2 and sell_c <= 1:    verdict, conf = 'BUY',         int(buy_c/total*100)
    elif sell_c >= 3:                   verdict, conf = 'STRONG SELL', int(sell_c/total*100)
    elif sell_c == 2 and buy_c <= 1:    verdict, conf = 'SELL',        int(sell_c/total*100)
    else:                               verdict, conf = 'HOLD',        int(hold_c/total*100) if hold_c else 50
    return verdict, conf, signals, reasons


# ─── Price Prediction ─────────────────────────────────────────────────────────
def predict_prices(df, days=14):
    close = df['Close'].values
    X = np.arange(len(close)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, close)
    future_X = np.arange(len(close), len(close) + days).reshape(-1, 1)
    predicted = model.predict(future_X)
    last_date  = df.index[-1]
    freq       = pd.tseries.frequencies.to_offset(pd.infer_freq(df.index) or 'D')
    future_dates = pd.date_range(start=last_date, periods=days + 1, freq=freq)[1:]
    r2 = model.score(X, close)
    return future_dates, predicted, r2


def plot_prediction(df, symbol):
    future_dates, predicted, r2 = predict_prices(df)
    cur, _ = get_currency(symbol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], name='Historical Price',
        line=dict(color='#6366f1', width=2)))
    fig.add_trace(go.Scatter(
        x=list(df.index) + list(future_dates),
        y=list(df['Close']) + list(predicted),
        name='Trend Line', line=dict(color='rgba(99,102,241,0.3)', width=1, dash='dot'),
        showlegend=False))
    fig.add_trace(go.Scatter(
        x=future_dates, y=predicted, name='Predicted (14 days)',
        line=dict(color='#f59e0b', width=2.5, dash='dash'),
        mode='lines+markers',
        marker=dict(size=5, color='#f59e0b', symbol='circle')))
    fig.add_vrect(
        x0=df.index[-1], x1=future_dates[-1],
        fillcolor='rgba(245,158,11,0.04)', line_width=0)
    fig.add_annotation(
        x=future_dates[len(future_dates)//2],
        y=max(predicted) * 1.01,
        text="◀ PREDICTION ZONE ▶",
        font=dict(color='#f59e0b', size=10, family='JetBrains Mono'),
        showarrow=False)
    fig.update_layout(
        height=320, margin=dict(t=20, b=20, l=8, r=8),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family='Space Grotesk', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#0f1623', bordercolor='#1e293b', font_color='#e2e8f0'))
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, tickprefix=cur)
    return fig, predicted, r2


# ─── Volatility ───────────────────────────────────────────────────────────────
def get_volatility(df):
    daily_returns = df['Close'].pct_change().dropna()
    vol = daily_returns.std() * np.sqrt(252) * 100
    if vol < 15:   label, color = "LOW",     "#10b981"
    elif vol < 30: label, color = "MEDIUM",  "#f59e0b"
    elif vol < 50: label, color = "HIGH",    "#ef4444"
    else:          label, color = "EXTREME", "#a855f7"
    return round(vol, 2), label, color


def plot_volatility_gauge(vol, label, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(vol, 80),
        number=dict(suffix="%", font=dict(color=color, size=32, family='JetBrains Mono')),
        gauge=dict(
            axis=dict(range=[0, 80], tickcolor='#334155',
                      tickfont=dict(color='#475569', size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor='#0c1018',
            bordercolor='#1e293b',
            steps=[
                dict(range=[0,  15], color='rgba(16,185,129,0.1)'),
                dict(range=[15, 30], color='rgba(245,158,11,0.1)'),
                dict(range=[30, 50], color='rgba(239,68,68,0.1)'),
                dict(range=[50, 80], color='rgba(168,85,247,0.1)'),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=min(vol,79))
        ),
        title=dict(text=f"<b>{label}</b>", font=dict(color=color, size=14, family='JetBrains Mono'))
    ))
    fig.update_layout(
        height=220, margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor='#0f1623', font=dict(color=FONT_COLOR))
    return fig


# ─── Charts ───────────────────────────────────────────────────────────────────
def plot_main_chart(df, symbol):
    c = dict(up='#10b981', down='#ef4444', ma20='#6366f1', ma50='#f59e0b',
             bb='rgba(99,102,241,0.1)', bb_line='rgba(99,102,241,0.35)',
             rsi='#38bdf8', vol_up='rgba(16,185,129,0.5)', vol_down='rgba(239,68,68,0.5)')
    cur, _ = get_currency(symbol)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.58, 0.22, 0.20])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', increasing=dict(line=dict(color=c['up'], width=1), fillcolor=c['up']),
        decreasing=dict(line=dict(color=c['down'], width=1), fillcolor=c['down'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color=c['bb_line'], width=1, dash='dot'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color=c['bb_line'], width=1, dash='dot'),
        fill='tonexty', fillcolor=c['bb'], showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA 20', line=dict(color=c['ma20'], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA 50', line=dict(color=c['ma50'], width=1.5)), row=1, col=1)
    vol_colors = [c['vol_up'] if cl >= op else c['vol_down'] for cl, op in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color=c['rsi'], width=2), showlegend=False), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.05)', line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor='rgba(16,185,129,0.05)', line_width=0, row=3, col=1)
    fig.add_hline(y=70, line_dash='dot', line_color='rgba(239,68,68,0.4)', line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash='dot', line_color='rgba(16,185,129,0.4)', line_width=1, row=3, col=1)
    fig.update_layout(height=680, margin=dict(t=16, b=16, l=8, r=8),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family='Space Grotesk', size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#0f1623', bordercolor='#1e293b', font_color='#e2e8f0'))
    fig.update_xaxes(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                     showspikes=True, spikecolor='#334155', spikethickness=1)
    fig.update_yaxes(gridcolor=GRID_COLOR, showgrid=True, zeroline=False, tickprefix=cur)
    return fig

def plot_macd_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#6366f1', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='#f59e0b', width=2)))
    hist_colors = ['rgba(16,185,129,0.5)' if v >= 0 else 'rgba(239,68,68,0.5)' for v in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color=hist_colors))
    fig.add_hline(y=0, line_color='#1e293b', line_width=1)
    fig.update_layout(height=240, margin=dict(t=10, b=10, l=8, r=8),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR, family='Space Grotesk', size=11),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        hovermode='x unified', hoverlabel=dict(bgcolor='#0f1623', bordercolor='#1e293b', font_color='#e2e8f0'))
    fig.update_xaxes(gridcolor=GRID_COLOR, showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, showgrid=True, zeroline=False)
    return fig


# ─── PDF Export ───────────────────────────────────────────────────────────────
def clean(text):
    """Replace all non-latin-1 characters so fpdf built-in fonts don't crash."""
    return (str(text)
            .replace('\u2014', '-')   # em dash  —  -> -
            .replace('\u2013', '-')   # en dash  –  -> -
            .replace('\u2019', "'")   # right single quote
            .replace('\u2018', "'")   # left single quote
            .replace('\u201c', '"')   # left double quote
            .replace('\u201d', '"')   # right double quote
            .replace('\u20b9', 'Rs.') # rupee sign  ₹  -> Rs.
            .replace('\u25b2', '^')   # up triangle
            .replace('\u25bc', 'v')   # down triangle
            .encode('latin-1', errors='replace').decode('latin-1'))

def c(text):   # short alias
    return clean(text)

def generate_pdf(symbol, price, change, chg_pct, verdict, confidence, signals,
                 reasons, rsi, ma20, ma50, vol, vol_label, predicted, currency_sym):
    # Use old-style ln=True approach — works on ALL fpdf2 versions without XPos/YPos issues
    cur = 'Rs.' if currency_sym == '₹' else currency_sym

    pdf = FPDF()
    pdf.add_page()

    # background
    pdf.set_fill_color(6, 9, 18)
    pdf.rect(0, 0, 210, 297, 'F')

    # ── Title ──
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(165, 180, 252)
    pdf.set_xy(10, pdf.get_y())
    pdf.cell(190, 14, c('StockSense AI - Stock Analysis Report'), align='C')
    pdf.ln(14)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(190, 7, c(f'Generated on {datetime.now().strftime("%d %b %Y, %I:%M %p")}'), align='C')
    pdf.ln(13)

    # ── Stock Info Box ──
    box_y = pdf.get_y()
    pdf.set_fill_color(15, 22, 35)
    pdf.set_draw_color(30, 58, 95)
    pdf.rect(10, box_y, 190, 28, 'FD')
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(226, 232, 240)
    pdf.set_xy(14, box_y + 5)
    pdf.cell(90, 8, c(symbol))
    chg_color = (16, 185, 129) if change >= 0 else (239, 68, 68)
    pdf.set_text_color(*chg_color)
    sign = '+' if change >= 0 else ''
    pdf.set_xy(104, box_y + 5)
    pdf.cell(96, 8, c(f'{cur}{price:,.2f}  {sign}{change:.2f} ({chg_pct:+.2f}%)'), align='R')
    pdf.set_xy(14, box_y + 16)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(190, 6, c(f'RSI: {rsi:.1f}   MA20: {cur}{ma20:.2f}   MA50: {cur}{ma50:.2f}'))
    pdf.ln(20)

    # ── Recommendation ──
    v_rgb = (16,185,129) if 'BUY' in verdict else (239,68,68) if 'SELL' in verdict else (245,158,11)
    rec_y = pdf.get_y()
    pdf.set_fill_color(15, 22, 35)
    pdf.set_draw_color(*v_rgb)
    pdf.rect(10, rec_y, 190, 22, 'FD')
    pdf.set_xy(14, rec_y + 5)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(*v_rgb)
    pdf.cell(100, 8, c(f'AI Verdict: {verdict}'))
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(148, 163, 184)
    pdf.set_xy(114, rec_y + 5)
    pdf.cell(86, 8, c(f'Confidence: {confidence}%'), align='R')
    pdf.ln(30)

    # ── Signals ──
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(51, 65, 85)
    pdf.set_x(10)
    pdf.cell(190, 6, 'INDICATOR SIGNALS')
    pdf.ln(8)
    for ind, sig in signals.items():
        s_rgb = (16,185,129) if sig=='BUY' else (239,68,68) if sig=='SELL' else (245,158,11)
        row_y = pdf.get_y()
        pdf.set_fill_color(12, 16, 24)
        pdf.set_draw_color(30, 41, 59)
        pdf.rect(10, row_y, 190, 14, 'FD')
        pdf.set_xy(14, row_y + 3)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*s_rgb)
        pdf.cell(22, 6, c(sig))
        pdf.set_xy(38, row_y + 3)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(44, 6, c(ind))
        pdf.set_xy(84, row_y + 3)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(116, 6, c(reasons[ind]))
        pdf.ln(16)
    pdf.ln(4)

    # ── Volatility ──
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(51, 65, 85)
    pdf.set_x(10)
    pdf.cell(190, 6, 'VOLATILITY')
    pdf.ln(8)
    vol_y = pdf.get_y()
    pdf.set_fill_color(12, 16, 24)
    pdf.set_draw_color(30, 41, 59)
    pdf.rect(10, vol_y, 190, 12, 'FD')
    pdf.set_xy(14, vol_y + 2)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(190, 6, c(f'Annualised Volatility: {vol}%   Level: {vol_label}'))
    pdf.ln(18)

    # ── Prediction ──
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(51, 65, 85)
    pdf.set_x(10)
    pdf.cell(190, 6, 'PRICE PREDICTION (14-DAY LINEAR REGRESSION)')
    pdf.ln(8)
    pred_y = pdf.get_y()
    pdf.set_fill_color(12, 16, 24)
    pdf.set_draw_color(30, 41, 59)
    pdf.rect(10, pred_y, 190, 16, 'FD')
    pdf.set_xy(14, pred_y + 2)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(148, 163, 184)
    trend = "Upward" if predicted[-1] > predicted[0] else "Downward"
    day7  = predicted[6] if len(predicted) > 6 else predicted[-1]
    pdf.cell(190, 5, c(f'Day 7: {cur}{day7:.2f}   Day 14: {cur}{predicted[-1]:.2f}   Trend: {trend}'))
    pdf.ln(7)
    pdf.set_xy(14, pdf.get_y())
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(190, 5, 'Disclaimer: For educational use only. Not financial advice.')
    pdf.ln(14)

    # ── Footer ──
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(30, 41, 59)
    pdf.set_x(10)
    pdf.cell(190, 6, 'StockSense AI  |  Built with Python + Streamlit  |  Educational Project', align='C')

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(tmp.name)
    return tmp.name


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px">
        <div class="sidebar-logo">📈 StockSense AI</div>
        <div class="sidebar-tag">// real-time analyzer v3.0</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("NAVIGATE", ["📊 Analyzer", "💼 Portfolio", "🔮 Prediction", "⚡ Volatility"], label_visibility="collapsed")

    st.divider()

    raw_symbol = st.text_input("STOCK SYMBOL", value="AAPL", placeholder="AAPL / TCS / ICICIBANK").upper().strip()

    if raw_symbol.endswith('.NS') or raw_symbol.endswith('.BO'):
        symbol = raw_symbol
        st.caption(f"🇮🇳 NSE/BSE detected")
    elif raw_symbol in INDIAN_STOCKS:
        symbol = raw_symbol + ".NS"
        st.success(f"🇮🇳 Auto → `{symbol}`")
    else:
        symbol = raw_symbol

    c1, c2 = st.columns(2)
    with c1:
        if st.button("AAPL 🇺🇸"):    symbol = "AAPL"
        if st.button("TSLA 🇺🇸"):    symbol = "TSLA"
        if st.button("NVDA 🇺🇸"):    symbol = "NVDA"
        if st.button("GOOGL 🇺🇸"):   symbol = "GOOGL"
    with c2:
        if st.button("TCS 🇮🇳"):        symbol = "TCS.NS"
        if st.button("RELIANCE 🇮🇳"):   symbol = "RELIANCE.NS"
        if st.button("ICICIBANK 🇮🇳"):  symbol = "ICICIBANK.NS"
        if st.button("INFY 🇮🇳"):       symbol = "INFY.NS"

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    period   = st.select_slider("TIME PERIOD", options=["1mo","3mo","6mo","1y","2y"], value="6mo")
    interval = st.selectbox("INTERVAL", ["1d","1h","30m"], index=0)
    auto_ref = st.toggle("Auto Refresh (5 min)", value=False)
    st.caption("⚠️ Educational use only. Not financial advice.")


# ─── Fetch Data ───────────────────────────────────────────────────────────────
currency_sym, currency_code = get_currency(symbol)

st.markdown(f"""
<div class="header-banner">
    <p class="header-title">StockSense AI</p>
    <p class="header-sub">// {page.split()[1].lower()} &nbsp;·&nbsp; {symbol} &nbsp;·&nbsp; {period} &nbsp;·&nbsp; {currency_code}</p>
</div>""", unsafe_allow_html=True)

with st.spinner(f"Fetching {symbol}..."):
    df, info = fetch_stock_data(symbol, period, interval)

if df is None or df.empty:
    st.error(f"❌ Could not fetch **{symbol}**.")
    st.info("💡 Indian stocks: `ICICIBANK.NS` · `HDFCBANK.NS`\n\nUS stocks: `AAPL` · `TSLA` · `NVDA`")
    st.stop()

df = calculate_indicators(df)
df.dropna(inplace=True)
if len(df) < 5:
    st.error("Not enough data. Try a longer period.")
    st.stop()

latest  = df.iloc[-1]
prev    = df.iloc[-2]
price   = latest['Close']
change  = price - prev['Close']
chg_pct = (change / prev['Close']) * 100
rsi_val = latest['RSI']
ma20_val = latest['MA20']
ma50_val = latest['MA50']
verdict, confidence, signals, reasons = generate_recommendation(df)
vol, vol_label, vol_color = get_volatility(df)
v_type  = 'buy' if 'BUY' in verdict else 'sell' if 'SELL' in verdict else 'hold'

# ── Metric bar (always shown) ──
st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">Current Price</div>
    <div class="metric-value">{fmt_price(price, symbol)}</div>
    <div class="{'metric-change-up' if change>=0 else 'metric-change-down'}">{'▲' if change>=0 else '▼'} {abs(change):.2f} ({chg_pct:+.2f}%)</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">RSI (14)</div>
    <div class="metric-value">{rsi_val:.1f}</div>
    <div style="color:#334155;font-size:11px;margin-top:3px">{'Oversold' if rsi_val<30 else 'Overbought' if rsi_val>70 else 'Neutral'}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">MA 20</div>
    <div class="metric-value">{fmt_price(ma20_val, symbol)}</div>
    <div style="color:#334155;font-size:11px;margin-top:3px">20-day avg</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">MA 50</div>
    <div class="metric-value">{fmt_price(ma50_val, symbol)}</div>
    <div style="color:#334155;font-size:11px;margin-top:3px">50-day avg</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Volatility</div>
    <div class="metric-value" style="color:{vol_color}">{vol}%</div>
    <div style="color:{vol_color};font-size:11px;margin-top:3px">{vol_label}</div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Analyzer":

    st.markdown("""<div class="section-header"><span class="section-title">🎯 &nbsp;AI Recommendation Engine</span><div class="section-line"></div></div>""", unsafe_allow_html=True)

    rec_col, sig_col = st.columns([1, 1.6], gap="large")
    with rec_col:
        st.markdown(f"""
        <div class="verdict-wrapper verdict-{v_type}">
            <div class="verdict-badge badge-{v_type}">AI Signal</div>
            <div class="verdict-text-{v_type}">{verdict}</div>
            <div style="color:#475569;font-size:12px;margin-top:10px;font-family:'JetBrains Mono',monospace">confidence score</div>
            <div class="conf-bar-bg"><div class="conf-bar-fill-{v_type}" style="width:{confidence}%"></div></div>
            <div style="color:#94a3b8;font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace">{confidence}%</div>
            <div style="color:#1e293b;font-size:11px;margin-top:12px">based on 4 technical indicators</div>
        </div>""", unsafe_allow_html=True)

    with sig_col:
        st.markdown("<div style='color:#334155;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px'>Indicator Breakdown</div>", unsafe_allow_html=True)
        for ind, sig in signals.items():
            st_type = 'buy' if sig=='BUY' else 'sell' if sig=='SELL' else 'hold'
            st.markdown(f"""
            <div class="signal-row">
                <div class="signal-dot dot-{st_type}"></div>
                <div>
                    <div style="display:flex;gap:10px;align-items:center">
                        <span class="signal-verdict-{st_type}">{sig}</span>
                        <span class="signal-name">{ind}</span>
                    </div>
                    <div class="signal-reason">{reasons[ind]}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="section-header"><span class="section-title">📊 &nbsp;Price · Bollinger Bands · Volume · RSI</span><div class="section-line"></div></div>""", unsafe_allow_html=True)
    st.plotly_chart(plot_main_chart(df, symbol), use_container_width=True)

    st.markdown("""<div class="section-header"><span class="section-title">📉 &nbsp;MACD Indicator</span><div class="section-line"></div></div>""", unsafe_allow_html=True)
    st.plotly_chart(plot_macd_chart(df), use_container_width=True)

    with st.expander("🗃️ Raw Data Table"):
        st.dataframe(df[['Open','High','Low','Close','Volume','MA20','MA50','RSI','MACD','MACD_Signal']].tail(30).round(2), use_container_width=True)

    # ── PDF Export button ──
    st.markdown("""<div class="section-header"><span class="section-title">📄 &nbsp;Export Report</span><div class="section-line"></div></div>""", unsafe_allow_html=True)
    _, predicted_vals, r2 = plot_prediction(df, symbol)
    if st.button("⬇️  Generate & Download PDF Report", use_container_width=True):
        with st.spinner("Generating PDF..."):
            pdf_path = generate_pdf(
                symbol, price, change, chg_pct, verdict, confidence,
                signals, reasons, rsi_val, ma20_val, ma50_val,
                vol, vol_label, predicted_vals, currency_sym
            )
            with open(pdf_path, 'rb') as f:
                st.download_button(
                    label="📄 Click here to download your report",
                    data=f.read(),
                    file_name=f"StockSense_{symbol}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            os.unlink(pdf_path)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💼 Portfolio":

    st.markdown("""<div class="section-header"><span class="section-title">💼 &nbsp;Portfolio Tracker</span><div class="section-line"></div></div>""", unsafe_allow_html=True)

    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []

    with st.expander("➕  Add Stock to Portfolio", expanded=len(st.session_state.portfolio) == 0):
        a1, a2, a3, a4 = st.columns([2,1,1,1])
        with a1: p_sym = st.text_input("Symbol", placeholder="e.g. TCS or AAPL").upper().strip()
        with a2: p_qty = st.number_input("Quantity", min_value=1, value=10)
        with a3: p_buy = st.number_input("Buy Price", min_value=0.01, value=100.0, format="%.2f")
        with a4:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            add_btn = st.button("Add ➕", use_container_width=True)

        if add_btn and p_sym:
            if p_sym in INDIAN_STOCKS and not p_sym.endswith('.NS'):
                p_sym = p_sym + ".NS"
            st.session_state.portfolio.append({'symbol': p_sym, 'qty': p_qty, 'buy_price': p_buy})
            st.success(f"✅ Added {p_sym}")
            st.rerun()

    if st.session_state.portfolio:
        total_invested = 0
        total_current  = 0
        rows_data = []

        for item in st.session_state.portfolio:
            cur_price = fetch_current_price(item['symbol'])
            if cur_price is None:
                cur_price = item['buy_price']
            cur_sym, _ = get_currency(item['symbol'])
            invested   = item['qty'] * item['buy_price']
            current    = item['qty'] * cur_price
            pnl        = current - invested
            pnl_pct    = (pnl / invested) * 100
            total_invested += invested
            total_current  += current
            rows_data.append({**item, 'cur_price': cur_price, 'invested': invested,
                               'current': current, 'pnl': pnl, 'pnl_pct': pnl_pct, 'cur_sym': cur_sym})

        total_pnl     = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested else 0
        pnl_col = "#10b981" if total_pnl >= 0 else "#ef4444"

        st.markdown(f"""
        <div class="port-summary">
            <div>
                <div class="metric-label">Total Invested</div>
                <div class="metric-value" style="font-size:16px">{total_invested:,.2f}</div>
            </div>
            <div>
                <div class="metric-label">Current Value</div>
                <div class="metric-value" style="font-size:16px">{total_current:,.2f}</div>
            </div>
            <div>
                <div class="metric-label">Total P&L</div>
                <div style="font-size:18px;font-weight:700;font-family:'JetBrains Mono',monospace;color:{pnl_col}">
                    {'▲' if total_pnl>=0 else '▼'} {abs(total_pnl):,.2f}
                </div>
            </div>
            <div>
                <div class="metric-label">Return %</div>
                <div style="font-size:18px;font-weight:700;font-family:'JetBrains Mono',monospace;color:{pnl_col}">
                    {total_pnl_pct:+.2f}%
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="portfolio-header">
            <div class="port-label">Symbol</div>
            <div class="port-label">Qty</div>
            <div class="port-label">Buy Price</div>
            <div class="port-label">Current</div>
            <div class="port-label">Invested</div>
            <div class="port-label">P&L</div>
        </div>""", unsafe_allow_html=True)

        for i, row in enumerate(rows_data):
            pnl_class = "port-profit" if row['pnl'] >= 0 else "port-loss"
            pnl_arrow = "▲" if row['pnl'] >= 0 else "▼"
            st.markdown(f"""
            <div class="portfolio-row">
                <div class="port-sym">{row['symbol']}</div>
                <div class="port-val">{row['qty']}</div>
                <div class="port-val">{row['cur_sym']}{row['buy_price']:.2f}</div>
                <div class="port-val">{row['cur_sym']}{row['cur_price']:.2f}</div>
                <div class="port-val">{row['cur_sym']}{row['invested']:,.2f}</div>
                <div class="{pnl_class}">{pnl_arrow} {row['cur_sym']}{abs(row['pnl']):,.2f} ({row['pnl_pct']:+.1f}%)</div>
            </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear Portfolio", use_container_width=False):
            st.session_state.portfolio = []
            st.rerun()

        # Portfolio pie chart
        if len(rows_data) > 1:
            st.markdown("""<div class="section-header"><span class="section-title">📊 &nbsp;Portfolio Allocation</span><div class="section-line"></div></div>""", unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(
                labels=[r['symbol'] for r in rows_data],
                values=[r['current'] for r in rows_data],
                hole=0.55,
                marker=dict(colors=['#6366f1','#10b981','#f59e0b','#38bdf8','#a855f7','#ef4444','#fb923c']),
                textfont=dict(color='#e2e8f0', size=12)))
            fig_pie.update_layout(
                height=320, paper_bgcolor=PAPER_BG,
                font=dict(color=FONT_COLOR, family='Space Grotesk'),
                legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
                margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No stocks in portfolio yet. Add some above!")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":

    st.markdown("""<div class="section-header"><span class="section-title">🔮 &nbsp;Price Prediction — Linear Regression Model</span><div class="section-line"></div></div>""", unsafe_allow_html=True)

    fig_pred, predicted_vals, r2 = plot_prediction(df, symbol)

    day7  = predicted_vals[6]  if len(predicted_vals) > 6  else predicted_vals[-1]
    day14 = predicted_vals[-1]
    trend = "Upward ▲" if day14 > price else "Downward ▼"
    trend_color = "#10b981" if day14 > price else "#ef4444"
    pct_change_14 = ((day14 - price) / price) * 100

    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(f"""<div class="pred-card"><div class="metric-label">Current Price</div><div class="metric-value">{fmt_price(price, symbol)}</div></div>""", unsafe_allow_html=True)
    p2.markdown(f"""<div class="pred-card"><div class="metric-label">Day 7 Target</div><div class="metric-value" style="color:#f59e0b">{fmt_price(day7, symbol)}</div></div>""", unsafe_allow_html=True)
    p3.markdown(f"""<div class="pred-card"><div class="metric-label">Day 14 Target</div><div class="metric-value" style="color:#f59e0b">{fmt_price(day14, symbol)}</div></div>""", unsafe_allow_html=True)
    p4.markdown(f"""<div class="pred-card"><div class="metric-label">Trend</div><div class="metric-value" style="color:{trend_color};font-size:16px">{trend}</div><div style="color:{trend_color};font-size:12px">{pct_change_14:+.2f}% in 14d</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown(f"""
    <div style="background:#0c1018;border:1px solid #1e293b;border-radius:12px;padding:16px 20px;margin-top:8px">
        <div style="color:#334155;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px">About This Model</div>
        <div style="color:#475569;font-size:13px;line-height:1.7">
            This prediction uses <strong style="color:#94a3b8">Linear Regression</strong> — a machine learning algorithm that fits a straight trend line
            through historical prices and extends it into the future. The model R² score is
            <strong style="color:#a5b4fc">{r2:.3f}</strong>
            (closer to 1.0 = better fit).<br><br>
            <strong style="color:#ef4444">⚠️ Important:</strong> Stock prices are influenced by news, earnings, global events — this model
            captures trend direction only. Always use alongside the technical indicators for a complete picture.
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Volatility":

    st.markdown("""<div class="section-header"><span class="section-title">⚡ &nbsp;Volatility Meter</span><div class="section-line"></div></div>""", unsafe_allow_html=True)

    v1, v2 = st.columns([1, 2], gap="large")

    with v1:
        st.markdown(f"""<div class="vol-gauge-wrap">
            <div style="color:#334155;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px">Annualised Volatility</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(plot_volatility_gauge(vol, vol_label, vol_color), use_container_width=True)
        st.markdown(f"""
        <div style="background:#0c1018;border:1px solid #1e293b;border-radius:10px;padding:14px;text-align:center">
            <div style="color:#475569;font-size:11px">What this means</div>
            <div style="color:{vol_color};font-size:13px;margin-top:6px;line-height:1.6">
                {'Low risk — price moves predictably. Suitable for conservative investors.' if vol_label=='LOW' else
                 'Moderate swings — normal for most stocks. Balanced risk.' if vol_label=='MEDIUM' else
                 'High risk — significant price swings. Only for risk-tolerant investors.' if vol_label=='HIGH' else
                 'Extreme risk — very unpredictable. Trade with maximum caution.'}
            </div>
        </div>""", unsafe_allow_html=True)

    with v2:
        # Daily returns distribution
        daily_returns = df['Close'].pct_change().dropna() * 100
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=daily_returns, nbinsx=40, name='Daily Returns',
            marker=dict(color='#6366f1', opacity=0.7,
                        line=dict(color='rgba(99,102,241,0.3)', width=0.5))))
        fig_dist.add_vline(x=0, line_color='#334155', line_width=1)
        fig_dist.add_vline(x=daily_returns.mean(), line_color='#10b981',
                           line_dash='dash', line_width=1.5,
                           annotation_text=f"Mean {daily_returns.mean():.2f}%",
                           annotation_font_color='#10b981')
        fig_dist.update_layout(
            title=dict(text='Daily Returns Distribution', font=dict(color='#64748b', size=12)),
            height=300, margin=dict(t=40,b=20,l=8,r=8),
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_COLOR, family='Space Grotesk', size=11),
            showlegend=False,
            hoverlabel=dict(bgcolor='#0f1623', bordercolor='#1e293b', font_color='#e2e8f0'))
        fig_dist.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, title='Daily Return %')
        fig_dist.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, title='Frequency')
        st.plotly_chart(fig_dist, use_container_width=True)

        # Stats table
        pos_days = (daily_returns > 0).sum()
        neg_days = (daily_returns < 0).sum()
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:4px">
            <div class="pred-card"><div class="metric-label">Avg Daily Return</div><div class="metric-value" style="font-size:16px;color:{'#10b981' if daily_returns.mean()>=0 else '#ef4444'}">{daily_returns.mean():+.2f}%</div></div>
            <div class="pred-card"><div class="metric-label">Best Day</div><div class="metric-value" style="font-size:16px;color:#10b981">+{daily_returns.max():.2f}%</div></div>
            <div class="pred-card"><div class="metric-label">Worst Day</div><div class="metric-value" style="font-size:16px;color:#ef4444">{daily_returns.min():.2f}%</div></div>
            <div class="pred-card"><div class="metric-label">Up / Down Days</div><div class="metric-value" style="font-size:14px"><span style="color:#10b981">{pos_days}</span> / <span style="color:#ef4444">{neg_days}</span></div></div>
        </div>""", unsafe_allow_html=True)


# Auto-refresh
if auto_ref:
    time.sleep(300)
    st.rerun()
