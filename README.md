# 📈 Market Minds | AI Financial Forecaster

> **Live App:** [market-minds-forecaster.streamlit.app](https://market-minds-forecaster.streamlit.app/)

A next-generation financial analytics dashboard that uses **Machine Learning** to forecast stock prices with confidence intervals, technical indicators, and a built-in backtesting engine — all in a clean, interactive UI.

---

## 🚀 Overview

**Market Minds** fetches real-time market data for any ticker (Stocks, Crypto, ETFs) and runs it through Meta's **Prophet** forecasting model to generate 30–365 day price predictions. The app is built entirely in Python and deployed on Streamlit Cloud.

---

## ⚡ Key Features

| Feature | Description |
|---|---|
| 🤖 **AI Forecasting** | Prophet additive regression model with configurable forecast horizon (30–365 days) and confidence intervals |
| 📊 **Interactive Charts** | Plotly-powered charts with zoom, pan, and hover analysis |
| 🕵️ **Backtesting Engine** | "Reality Check" tab hides recent data to test model accuracy against actual performance |
| 🕯️ **Technical Analysis** | Candlestick charts, 50-Day & 200-Day SMA, RSI indicators |
| ⚡ **Live Data** | Real-time market data via Yahoo Finance API for any global ticker |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Frontend** | Streamlit (Python) |
| **ML Engine** | Prophet (Meta/Facebook) |
| **Visualization** | Plotly, Plotly Express |
| **Data Source** | Yahoo Finance API (`yfinance`) |
| **Data Processing** | Pandas, NumPy |

---

## 📁 Project Structure

```
market-minds-forecaster/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📦 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Nielr2004/Market-Minds-Forecaster.git
   cd Market-Minds-Forecaster
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

---

## 🌐 Live Demo

The app is deployed and publicly accessible at:
**[https://market-minds-forecaster.streamlit.app/](https://market-minds-forecaster.streamlit.app/)**

---

## 👤 Author

**Snehashis Roy**
- 📧 roysnehashis2004@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/your-linkedin)
- 🐙 [GitHub](https://github.com/Nielr2004)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
