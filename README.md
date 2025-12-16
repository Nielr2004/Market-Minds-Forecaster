# 📈 Market Minds | AI Financial Forecaster

### 🚀 Overview
**Market Minds** is a next-generation financial analytics tool that leverages **Machine Learning** to predict future stock prices. Built with **Streamlit** for the frontend and **Meta's Prophet** for the backend, it offers real-time forecasting, interactive charts, and technical analysis indicators (SMA, RSI) in a clean, professional dashboard.

### ⚡ Key Features
* **🤖 AI Forecasting:** Uses the `Prophet` additive regression model to generate 30-365 day price predictions with confidence intervals.
* **📊 Interactive Charts:** Seamless, transparent `Plotly` graphs that allow users to zoom, pan, and analyze specific data points.
* **🕵️‍♂️ Backtesting Engine:** A built-in "Reality Check" tab that hides recent data to test the model's accuracy against actual market performance.
* **🕯️ Technical Analysis:** Includes Candlestick charts and Moving Averages (50-Day & 200-Day SMA) for deeper market insight.
* **⚡ Live Data:** Fetches real-time market data for any ticker (Stocks, Crypto, ETFs) using `yfinance`.

### 🛠️ Tech Stack
* **Frontend:** Streamlit (Python)
* **ML Engine:** Prophet (by Meta/Facebook)
* **Visualization:** Plotly & Plotly Express
* **Data Source:** Yahoo Finance API (`yfinance`)
* **Data Processing:** Pandas & NumPy

### 📸 Preview
*(You can upload a screenshot of your app here later!)*

### 📦 How to Run Locally
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Market-Minds-Forecaster.git](https://github.com/YOUR_USERNAME/Market-Minds-Forecaster.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    streamlit run app.py
    ```
