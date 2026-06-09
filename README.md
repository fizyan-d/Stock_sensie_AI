<div align="center">

# 📈 StockSense AI

### Real-Time Stock Market Analyzer + AI Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=for-the-badge&logo=streamlit)
![yfinance](https://img.shields.io/badge/yfinance-Live%20Data-green?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

*A Python + Data Science mini project that fetches real-time stock data, analyzes it using 4 trading algorithms, and gives intelligent BUY / SELL / HOLD recommendations.*

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Real-Time Data** | Fetches live stock prices from Yahoo Finance via `yfinance` |
| 🤖 **AI Recommendation** | 4 algorithms vote → combined BUY/SELL/HOLD with confidence % |
| 📈 **Interactive Charts** | Candlestick, Bollinger Bands, Volume, RSI, MACD via Plotly |
| 🔮 **Price Prediction** | 14-day forecast using Linear Regression (scikit-learn) |
| 💼 **Portfolio Tracker** | Track multiple stocks with live P&L and allocation chart |
| ⚡ **Volatility Meter** | Speedometer gauge + daily returns distribution histogram |
| 📄 **PDF Export** | Download full analysis report as a formatted PDF |
| 🇮🇳 **Indian Stocks** | Auto-detects NSE stocks — type `TCS` instead of `TCS.NS` |
| 💱 **Smart Currency** | Shows ₹ for Indian stocks, $ for US stocks automatically |

---

## 🧠 Technical Indicators Used

### 1. Moving Average Crossover
- **20-day MA** vs **50-day MA**
- Golden Cross (MA20 > MA50) → 📗 BUY
- Death Cross (MA20 < MA50) → 📕 SELL

### 2. RSI — Relative Strength Index
- RSI < 30 → Oversold → 📗 BUY
- RSI > 70 → Overbought → 📕 SELL
- 30–70 → 🟡 HOLD

### 3. MACD — Moving Average Convergence Divergence
- MACD crosses above Signal → 📗 BUY
- MACD crosses below Signal → 📕 SELL

### 4. Bollinger Bands
- Price at lower band → 📗 BUY
- Price at upper band → 📕 SELL
- Price inside bands → 🟡 HOLD

### Final Verdict = Majority Vote
```
3-4 BUY  votes → STRONG BUY  (🟢)
2   BUY  votes → BUY         (🟢)
3-4 SELL votes → STRONG SELL (🔴)
2   SELL votes → SELL        (🔴)
else           → HOLD        (🟡)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or above
- pip package manager

### Installation

**Step 1 — Clone this repository**
```bash
git clone https://github.com/YOUR_USERNAME/stocksense-ai.git
cd stocksense-ai
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
pip install pandas-ta --no-build-isolation
```

**Step 3 — Run the app**
```bash
python -m streamlit run app.py
```

**Step 4 — Open in browser**
```
http://localhost:8501
```

---

## 📦 Dependencies

```
streamlit       — Web application framework
yfinance        — Real-time stock data from Yahoo Finance
pandas          — Data manipulation and analysis
pandas-ta       — Technical analysis indicators (RSI, MACD, BB)
plotly          — Interactive charts and visualizations
scikit-learn    — Linear Regression for price prediction
numpy           — Numerical computations
fpdf2           — PDF report generation
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🎮 Usage Guide

### Analyzer Page
1. Type a stock symbol in the sidebar (e.g. `AAPL`, `TCS`, `ICICIBANK`)
2. Select time period (1mo to 2y) and interval (daily/hourly)
3. View the AI recommendation with confidence score
4. Explore interactive charts
5. Click **Generate PDF Report** to download analysis

### Portfolio Page
1. Click **Add Stock to Portfolio**
2. Enter symbol, quantity, and buy price
3. View live P&L for each stock
4. See total portfolio returns and allocation chart

### Prediction Page
- View 14-day price forecast with prediction zone on chart
- See Day 7 and Day 14 price targets
- Check R² score for model accuracy

### Volatility Page
- Speedometer gauge showing LOW/MEDIUM/HIGH/EXTREME risk
- Daily returns histogram
- Best day, worst day, average return statistics

---

## 🇮🇳 Indian Stock Symbols

| Company | Symbol | Auto-detected? |
|---------|--------|----------------|
| TCS | TCS.NS | ✅ Type just `TCS` |
| Infosys | INFY.NS | ✅ Type just `INFY` |
| Reliance | RELIANCE.NS | ✅ Type just `RELIANCE` |
| HDFC Bank | HDFCBANK.NS | ✅ Type just `HDFCBANK` |
| ICICI Bank | ICICIBANK.NS | ✅ Type just `ICICIBANK` |
| Wipro | WIPRO.NS | ✅ Type just `WIPRO` |
| Nifty 50 | ^NSEI | Manual |

## 🇺🇸 US Stock Symbols

| Company | Symbol |
|---------|--------|
| Apple | AAPL |
| Tesla | TSLA |
| NVIDIA | NVDA |
| Google | GOOGL |
| Microsoft | MSFT |
| Amazon | AMZN |

---

## 🗂️ Project Structure

```
stocksense-ai/
├── app.py                  ← Main application (all code)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── PROJECT_REPORT.md       ← Detailed project documentation
└── VIVA_QUESTIONS.md       ← 50 Q&A for exam preparation
```

---

## 🛠️ Tech Stack

```
Language    : Python 3.9+
Frontend    : Streamlit + Custom CSS
Charts      : Plotly (interactive)
Data Source : Yahoo Finance (via yfinance)
ML Model    : Linear Regression (scikit-learn)
PDF Engine  : fpdf2
Indicators  : pandas-ta
```

---

## ⚠️ Disclaimer

This project is built **for educational purposes only** as part of a university Data Science course.

> The recommendations generated by this application are based on mathematical indicators and **do not constitute financial advice**. Do not make real investment decisions based on this tool. Always consult a certified financial advisor before investing.

---

## 👨‍💻 Author

Built with ❤️ using Python | University Mini Project — Python in Data Science

---

<div align="center">
⭐ If you found this project helpful, please give it a star!
</div>
