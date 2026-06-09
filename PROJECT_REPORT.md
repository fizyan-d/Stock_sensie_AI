# StockSense AI — Complete Project Report

## Python in Data Science | Mini Project Report

---

## 1. PROJECT OVERVIEW

**Project Name:** StockSense AI — Real-Time Stock Market Analyzer  
**Technology:** Python, Streamlit, Machine Learning  
**Domain:** Financial Data Science  

### What is this project?
StockSense AI is a real-time stock market analysis web application built entirely in Python. It fetches live stock price data from the internet, applies four technical trading algorithms to analyze the stock, and gives the user an intelligent BUY / SELL / HOLD recommendation with a confidence score. It also features a Portfolio Tracker, a 14-day Price Prediction model using Linear Regression, a Volatility Meter, and a PDF report export.

---

## 2. PROBLEM STATEMENT

Stock market investing is complex. Individual investors often don't know when to buy or sell a stock. Professional traders use technical indicators like RSI, MACD, Moving Averages, and Bollinger Bands — but these require deep knowledge to interpret. This project solves that problem by automating the analysis and combining all indicators into one simple recommendation.

---

## 3. OBJECTIVES

- Fetch real-time stock market data using Python
- Apply 4 technical analysis algorithms to generate trading signals
- Combine signals into a unified AI recommendation engine
- Visualize data using interactive professional charts
- Add a portfolio tracker to monitor multiple stocks
- Predict future prices using Linear Regression (Machine Learning)
- Measure and display stock volatility
- Export a complete analysis report as PDF

---

## 4. TECH STACK & LIBRARIES

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | Latest | Web application framework — creates the entire UI |
| `yfinance` | Latest | Fetches real-time and historical stock data from Yahoo Finance |
| `pandas` | Latest | Data manipulation — stores stock data in DataFrames |
| `pandas-ta` | Latest | Technical Analysis library — calculates RSI, MACD, Bollinger Bands |
| `plotly` | Latest | Interactive charts — candlestick, line, bar, gauge charts |
| `scikit-learn` | Latest | Machine learning — Linear Regression for price prediction |
| `numpy` | Latest | Numerical calculations — arrays, math operations |
| `fpdf2` | Latest | PDF generation — exports analysis report |

### Why these libraries?

- **yfinance** is the most popular free Python library for fetching Yahoo Finance stock data. It supports thousands of stocks globally including NSE Indian stocks.
- **pandas** is the industry standard for data manipulation in Python. Stock data naturally fits into a tabular DataFrame structure.
- **pandas-ta** is a dedicated technical analysis library with 130+ indicators built-in, making RSI and MACD calculation a single line of code.
- **plotly** produces interactive charts that users can zoom, pan, and hover over — far superior to static matplotlib charts for a web app.
- **streamlit** converts a Python script into a full web app without any HTML/CSS/JavaScript knowledge needed.
- **scikit-learn** is the most popular ML library in Python. Linear Regression is one of its simplest and most interpretable models.

---

## 5. DATA SOURCE

**Source:** Yahoo Finance via the `yfinance` Python library  
**No static dataset is used** — the project fetches live data every time  
**Data fetched:** Open, High, Low, Close prices + Volume for each day/hour  
**Coverage:** US stocks (AAPL, TSLA, NVDA etc.) and Indian NSE stocks (TCS.NS, RELIANCE.NS etc.)  
**Update frequency:** Cached for 5 minutes, then re-fetched automatically  

---

## 6. PROJECT ARCHITECTURE

```
User opens browser (localhost:8501)
         |
   Streamlit Web App (app.py)
         |
    ┌────┴────────────────────────────┐
    │          4 Pages / Tabs         │
    ├─────────────────────────────────┤
    │  📊 Analyzer   │  💼 Portfolio  │
    │  🔮 Prediction │  ⚡ Volatility  │
    └────────────────┬────────────────┘
                     |
         ┌───────────┴──────────┐
         │    Core Functions    │
         ├──────────────────────┤
         │ fetch_stock_data()   │  ← yfinance
         │ calculate_indicators │  ← pandas-ta
         │ generate_recommendation() │ ← custom logic
         │ predict_prices()     │  ← scikit-learn
         │ get_volatility()     │  ← numpy
         │ generate_pdf()       │  ← fpdf2
         └──────────────────────┘
```

---

## 7. FEATURES IN DETAIL

### 7.1 Real-Time Data Fetching
The `yfinance` library fetches stock data by calling Yahoo Finance's API internally. The `fetch_stock_data()` function accepts a symbol (e.g. "AAPL"), a period (e.g. "6mo") and an interval (e.g. "1d"). It returns a pandas DataFrame with columns: Open, High, Low, Close, Volume.

Data is cached using `@st.cache_data(ttl=300)` — this means the same data is reused for 5 minutes to avoid excessive API calls.

### 7.2 Technical Indicators (calculate_indicators function)

**Moving Averages (MA20, MA50)**
- Rolling average of closing prices over last 20 and 50 days
- Calculated using `pandas .rolling(window=20).mean()`
- Smooths out price noise to show the underlying trend

**RSI — Relative Strength Index**
- Formula: RSI = 100 - (100 / (1 + RS)) where RS = avg gain / avg loss over 14 days
- Calculated using `ta.rsi(df['Close'], length=14)`
- Measures momentum — how fast prices are rising or falling

**MACD — Moving Average Convergence Divergence**
- Uses three values: MACD line (12-day EMA minus 26-day EMA), Signal line (9-day EMA of MACD), Histogram (MACD minus Signal)
- Calculated using `ta.macd(df['Close'], fast=12, slow=26, signal=9)`
- EMA = Exponential Moving Average — gives more weight to recent prices

**Bollinger Bands**
- Upper Band = 20-day MA + (2 × standard deviation)
- Lower Band = 20-day MA - (2 × standard deviation)
- Calculated using `ta.bbands(df['Close'], length=20, std=2)`
- Creates a "channel" around the price — extremes signal reversals

### 7.3 AI Recommendation Engine (generate_recommendation function)

This is the most important part of the project. Each of the 4 indicators independently votes BUY, SELL, or HOLD based on these rules:

| Indicator | BUY Condition | SELL Condition | HOLD Condition |
|-----------|--------------|----------------|----------------|
| MA Crossover | MA20 > MA50 | MA20 < MA50 | — |
| RSI | RSI < 30 (oversold) | RSI > 70 (overbought) | 30 ≤ RSI ≤ 70 |
| MACD | MACD > Signal line | MACD < Signal line | — |
| Bollinger Bands | Price at lower band | Price at upper band | Price inside bands |

**Final Verdict Logic (voting system):**
- 3 or 4 BUY votes → STRONG BUY
- 2 BUY votes → BUY
- 3 or 4 SELL votes → STRONG SELL
- 2 SELL votes → SELL
- Else → HOLD

Confidence score = (votes for verdict / 4) × 100%

This is a rule-based AI system — it uses domain knowledge encoded as rules rather than training on labeled data.

### 7.4 Price Prediction (predict_prices function)
- Uses `LinearRegression` from scikit-learn
- Input features (X): day numbers (0, 1, 2, 3, ...)
- Target (y): closing prices
- After training on historical data, it predicts the next 14 future days
- R² score measures how well the line fits the historical data (0 to 1)
- Limitation: assumes price follows a linear trend — does not account for sudden news events

### 7.5 Volatility Meter (get_volatility function)
- Calculates daily returns: `df['Close'].pct_change()`
- Annualised volatility: `daily_returns.std() × √252 × 100`
- 252 = number of trading days in a year
- Scale: LOW (<15%), MEDIUM (15-30%), HIGH (30-50%), EXTREME (>50%)
- Displayed using a Plotly gauge/speedometer chart

### 7.6 Portfolio Tracker
- Stocks stored in `st.session_state.portfolio` (Streamlit's in-memory storage)
- For each stock: fetches current live price using yfinance
- Calculates: Invested = Qty × Buy Price, Current = Qty × Current Price
- P&L = Current − Invested, Return% = (P&L / Invested) × 100
- Donut chart shows allocation percentage across all stocks

### 7.7 PDF Export (generate_pdf function)
- Uses `fpdf2` library to generate a PDF from scratch
- Creates dark-themed boxes, colored text, tables using drawing commands
- All Unicode characters (—, ₹, ▲) are cleaned to ASCII equivalents
- Saves to a temporary file, then serves as a download button in Streamlit

---

## 8. KEY ALGORITHMS EXPLAINED

### Linear Regression
Linear Regression finds the best-fit straight line through data points by minimizing the sum of squared errors (MSE — Mean Squared Error). The equation is:

```
y = mx + b
where:
  y = predicted price
  x = day number
  m = slope (rate of price change)
  b = y-intercept
```

Scikit-learn's `LinearRegression().fit(X, y)` calculates m and b automatically. `.predict(future_X)` then extends the line into the future.

### RSI Formula
```
Step 1: Calculate daily price changes
Step 2: Separate gains and losses
Step 3: Average Gain = mean of gains over 14 days
Step 4: Average Loss = mean of losses over 14 days
Step 5: RS = Average Gain / Average Loss
Step 6: RSI = 100 - (100 / (1 + RS))
```

### Volatility Formula
```
Daily Return = (Today's Price - Yesterday's Price) / Yesterday's Price
Daily Std Dev = standard deviation of all daily returns
Annualised Volatility = Daily Std Dev × √252 × 100
```

---

## 9. HOW TO RUN THE PROJECT

### Requirements
- Python 3.9 or above
- Windows / Mac / Linux

### Step-by-step
```bash
# 1. Navigate to project folder
cd Desktop/stock_analyzer

# 2. Install all libraries
pip install -r requirements.txt

# 3. Install pandas-ta separately
pip install pandas-ta --no-build-isolation

# 4. Run the app
python -m streamlit run app.py
```

Open browser at: `http://localhost:8501`

---

## 10. PROJECT FILE STRUCTURE

```
stock_analyzer/
├── app.py              ← Entire application (single file)
├── requirements.txt    ← All library dependencies
├── README.md           ← GitHub documentation
├── PROJECT_REPORT.md   ← This report
└── VIVA_QUESTIONS.md   ← Viva preparation
```

---

## 11. LIMITATIONS

1. **Linear Regression is too simple** for real stock prediction — real models use LSTM neural networks
2. **No sentiment analysis** — news and social media heavily influence stocks
3. **Not real-time tick data** — updates every 5 minutes, not millisecond level
4. **Rule-based AI** — does not learn from data, uses fixed rules
5. **Portfolio data is not persistent** — clears when you refresh the browser

---

## 12. FUTURE IMPROVEMENTS

1. Add LSTM (Long Short-Term Memory) neural network for better prediction
2. Add news sentiment analysis using NLP
3. Connect to a database to save portfolio permanently
4. Add email/SMS price alerts
5. Add sector heatmap for Nifty 50

---

## 13. CONCLUSION

StockSense AI successfully demonstrates the application of Python and Data Science in real-world financial analysis. It combines data fetching, data processing, technical analysis algorithms, machine learning, interactive visualization, and PDF reporting in a single cohesive application. The project covers core data science concepts including feature engineering, statistical analysis, regression modeling, and data visualization.
