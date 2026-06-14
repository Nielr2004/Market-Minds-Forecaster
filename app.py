import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import os

# 1. PAGE SETUP
ROOT_DIR = os.path.dirname(__file__)
icon_path = os.path.join(ROOT_DIR, "chart_icon.svg")

st.set_page_config(
    page_title="Market Minds | Forecasting Dashboard",
    page_icon=icon_path,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CLEAN UI STYLING
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
    }
    .stApp {
        color: #e2e8f0;
        background-color: #0f172a;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 14px;
        border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.95rem;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #334155;
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #cbd5e1;
        font-size: 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8;
        border-bottom: 2px solid #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# 3. SIDEBAR CONTROLS
st.sidebar.markdown("## Market Minds")
st.sidebar.caption("Financial forecasting dashboard")
st.sidebar.markdown("---")

selected_stock = st.sidebar.text_input("Ticker symbol", "BTC-USD").upper()
n_years = st.sidebar.slider("Training data (years)", 1, 5, 2)
forecast_days = st.sidebar.slider("Forecast horizon (days)", 30, 365, 90)

st.sidebar.markdown("### Advanced tuning")
changepoint_scale = st.sidebar.slider(
    "Trend sensitivity",
    0.01,
    0.5,
    0.05,
    help="Higher values fit recent moves more closely. Lower values produce a smoother trend."
)
seasonality_mode = st.sidebar.selectbox(
    "Seasonality mode",
    ["additive", "multiplicative"],
    index=1
)

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.experimental_rerun()

# 4. CHART HELPER FUNCTION

def style_plot(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cfcfcf'),
        xaxis=dict(showgrid=False, color='#666'),
        yaxis=dict(showgrid=True, gridcolor='#222', color='#666'),
        margin=dict(l=0, r=0, t=20, b=0),
        hovermode='x unified',
        height=540
    )
    return fig

# 5. DATA ENGINE
@st.cache_data(ttl=3600)
def load_data(ticker, years):
    try:
        start = (date.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        end = date.today().strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start, end=end)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.reset_index(inplace=True)
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        return data
    except Exception:
        return pd.DataFrame()


def format_currency(value):
    return f"${value:,.2f}"


def add_layout_spacing(height=18):
    st.markdown(f"<div style='height: {height}px'></div>", unsafe_allow_html=True)


def render_market_header(ticker, years, horizon, price, delta, pct_change):
    st.title(ticker)
    st.markdown("#### Market snapshot")

    with st.container():
        left, right = st.columns([2.5, 1])
        with left:
            st.markdown(
                f"**Market view for {ticker}**  \n"
                f"{years} years of historical prices and {horizon}-day forecast horizon."
            )
        with right:
            color_hex = "#00FFA3" if delta > 0 else "#f97316"
            st.markdown(
                f"<div style='text-align: right;'>"
                f"<div style='font-size: 32px; font-weight: 700;'>{format_currency(price)}</div>"
                f"<div style='color: {color_hex}; font-size: 16px;'>{delta:+.2f} ({pct_change:+.2f}%)</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    add_layout_spacing(20)


def render_indicators(data):
    with st.container():
        st.subheader("Key indicators")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Highest in period", format_currency(data['High'].max()))
        metric_cols[1].metric("Lowest in period", format_currency(data['Low'].min()))
        metric_cols[2].metric("Latest volume", f"{int(data.iloc[-1]['Volume']):,}")
        metric_cols[3].metric("50-day moving average", format_currency(data.iloc[-1]['SMA_50']))

    add_layout_spacing(18)


def render_forecast_tab(data, forecast_days, current_price):
    st.subheader("Forecast summary")
    st.write("Forecast applies historical trend and seasonality to estimate future movement.")

    with st.spinner("Preparing forecast..."):
        df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
        model = Prophet(changepoint_prior_scale=changepoint_scale, seasonality_mode=seasonality_mode)
        model.fit(df_train)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        fig_p = plot_plotly(model, forecast)
        fig_p.data[2].line.color = '#4F8BF9'
        fig_p.data[0].marker.color = 'rgba(79, 139, 249, 0.15)'
        fig_p.data[1].marker.color = 'rgba(79, 139, 249, 0.15)'
        st.plotly_chart(style_plot(fig_p), use_container_width=True)

        pred_price = forecast.iloc[-1]['yhat']
        trend = "Bullish" if pred_price > current_price else "Bearish"

        result_cols = st.columns([3, 1])
        result_cols[0].markdown(f"**Forecast target ({forecast_days} days):** {format_currency(pred_price)}")
        if trend == "Bullish":
            result_cols[1].success("Trend: Bullish")
        else:
            result_cols[1].error("Trend: Bearish")


def render_backtest_tab(data, forecast_days):
    st.subheader("Backtest accuracy")
    st.write("The model is validated against a held-out historical period.")

    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    train_len = len(df_train) - forecast_days

    if train_len < 30:
        st.error("Not enough history for a reliable backtest.")
        return

    train_set = df_train.iloc[:train_len]
    test_set = df_train.iloc[train_len:]
    model = Prophet(changepoint_prior_scale=changepoint_scale, seasonality_mode=seasonality_mode)
    model.fit(train_set)
    future_bt = model.make_future_dataframe(periods=forecast_days)
    forecast_bt = model.predict(future_bt)

    combined = pd.merge(test_set, forecast_bt[['ds', 'yhat']], on='ds')
    combined['AbsError'] = (combined['y'] - combined['yhat']).abs()
    mae = combined['AbsError'].mean()
    mape = (combined['AbsError'] / combined['y']).mean() * 100

    summary_cols = st.columns(3)
    summary_cols[0].metric("Mean absolute error", format_currency(mae))
    summary_cols[1].metric("MAPE", f"{mape:.2f}%")
    if mape < 5:
        summary_cols[2].success("Excellent accuracy")
    elif mape < 10:
        summary_cols[2].warning("Good accuracy")
    else:
        summary_cols[2].error("Limited accuracy")

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=train_set['ds'], y=train_set['y'], name="Training data", line=dict(color='#888')))
    fig_bt.add_trace(go.Scatter(x=test_set['ds'], y=test_set['y'], name="Actual price", line=dict(color='#00FFA3', width=2)))
    fig_bt.add_trace(go.Scatter(x=forecast_bt['ds'], y=forecast_bt['yhat'], name="Forecast", line=dict(color='#f97316', dash='dot')))

    st.plotly_chart(style_plot(fig_bt), use_container_width=True)


def render_market_chart_tab(data, ticker):
    st.subheader("Market chart")
    st.write("Candlestick view with moving averages highlights market structure.")

    fig_candle = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'],
        name=ticker
    )])
    fig_candle.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name="50 SMA", line=dict(color='orange', width=1)))
    fig_candle.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], name="200 SMA", line=dict(color='purple', width=1)))
    fig_candle.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(style_plot(fig_candle), use_container_width=True)


# 6. MAIN APP
data = load_data(selected_stock, n_years)

if data.empty:
    st.warning(f"Could not validate ticker '{selected_stock}'. Please try again.")
else:
    current_price = data.iloc[-1]['Close']
    prev_price = data.iloc[-2]['Close']
    delta = current_price - prev_price
    pct_change = (delta / prev_price) * 100

    render_market_header(selected_stock, n_years, forecast_days, current_price, delta, pct_change)
    render_indicators(data)

    tab_forecast, tab_backtest, tab_chart = st.tabs(["Forecast", "Backtest", "Market chart"])

    with tab_forecast:
        render_forecast_tab(data, forecast_days, current_price)

    with tab_backtest:
        render_backtest_tab(data, forecast_days)

    with tab_chart:
        render_market_chart_tab(data, selected_stock)