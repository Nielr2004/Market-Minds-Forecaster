import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# 1. PAGE SETUP
st.set_page_config(
    page_title="Market Minds | AI Analyst", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. MODERN UI STYLING (CSS)
st.markdown("""
<style>
    /* Remove top padding to maximize screen space */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }

    /* METRIC CARDS: Clean, minimal, no heavy borders */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05); /* Subtle glass effect */
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
        transition: 0.3s;
    }
    div[data-testid="stMetric"]:hover {
        background-color: rgba(255, 255, 255, 0.1); /* Light up on hover */
        border-color: rgba(255, 255, 255, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #a0a0a0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 600;
        color: #ffffff;
    }

    /* TABS: Sleek underline style instead of boxy buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 25px;
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border: none;
        color: #666;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #4F8BF9; /* Professional Blue */
        border-bottom: 2px solid #4F8BF9;
    }

    /* SIDEBAR: Darker and cleaner */
    section[data-testid="stSidebar"] {
        background-color: #111;
    }
</style>
""", unsafe_allow_html=True)

# 3. SIDEBAR CONTROLS
st.sidebar.header("📊 Market Minds")
st.sidebar.caption("AI-Powered Financial Forecasting")
st.sidebar.markdown("---")

selected_stock = st.sidebar.text_input("Ticker Symbol", "BTC-USD").upper()
n_years = st.sidebar.slider("Training Data (Years)", 1, 5, 2)
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 365, 90)

st.sidebar.markdown("### ⚙️ Advanced Tuning")
changepoint_scale = st.sidebar.slider("Trend Sensitivity", 0.01, 0.5, 0.05, help="Higher = More flexible trend (fits noise). Lower = Smoother trend.")
seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ["additive", "multiplicative"], index=1)

if st.sidebar.button("↻ Clear Cache"):
    st.cache_data.clear()

# 4. CHART HELPER FUNCTION (Makes plots look seamless)
def style_plot(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cfcfcf'),
        xaxis=dict(showgrid=False, color='#666'),
        yaxis=dict(showgrid=True, gridcolor='#222', color='#666'),
        margin=dict(l=0, r=0, t=10, b=0), # Remove wasted margins
        hovermode="x unified",
        height=550
    )
    return fig

# 5. DATA ENGINE
@st.cache_data(ttl=3600)
def load_data(ticker, years):
    try:
        start = (date.today() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        end = date.today().strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start, end=end)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data.reset_index(inplace=True)
        # Add SMA
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        return data
    except Exception:
        return pd.DataFrame()

# 6. MAIN APP
data = load_data(selected_stock, n_years)

if data.empty:
    st.warning(f"⚠️ Could not validate ticker '{selected_stock}'. Please try again.")
else:
    # --- HEADER SECTION ---
    current_price = data.iloc[-1]['Close']
    prev_price = data.iloc[-2]['Close']
    delta = current_price - prev_price
    pct_change = (delta / prev_price) * 100
    
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.markdown(f"<h1 style='margin-bottom: 0px;'>{selected_stock}</h1>", unsafe_allow_html=True)
        st.caption(f"Analyzing last {n_years} years of market data • Forecast for next {forecast_days} days")
    with col_head2:
        # Custom HTML for big price display
        color_hex = "#00FFA3" if delta > 0 else "#FF4B4B"
        st.markdown(f"""
            <div style="text-align: right; padding-right: 10px;">
                <div style="font-size: 36px; font-weight: bold;">${current_price:,.2f}</div>
                <div style="color: {color_hex}; font-size: 18px;">{delta:+.2f} ({pct_change:+.2f}%)</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("") # Spacer

    # --- KPI GRID ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Highest (Period)", f"${data['High'].max():,.2f}")
    kpi2.metric("Lowest (Period)", f"${data['Low'].min():,.2f}")
    kpi3.metric("Volume (24h)", f"{data.iloc[-1]['Volume']:,}")
    kpi4.metric("50-Day SMA", f"${data.iloc[-1]['SMA_50']:,.2f}")

    st.write("") # Spacer

    # --- TABS ---
    tab_forecast, tab_backtest, tab_chart = st.tabs(["🔮 AI Forecast", "⚖️ Backtest Accuracy", "🕯️ Candlestick Chart"])

    # TAB 1: FORECAST
    with tab_forecast:
        with st.spinner("🤖 Crunching numbers..."):
            # Prophet Setup
            df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
            m = Prophet(changepoint_prior_scale=changepoint_scale, seasonality_mode=seasonality_mode)
            m.fit(df_train)
            future = m.make_future_dataframe(periods=forecast_days)
            forecast = m.predict(future)

            # Plot
            fig_p = plot_plotly(m, forecast)
            
            # Custom Styling for "TradingView" look
            fig_p.data[2].line.color = '#4F8BF9' # Main line (Blue)
            fig_p.data[0].marker.color = 'rgba(79, 139, 249, 0.15)' # Uncertainty band (Transparent Blue)
            fig_p.data[1].marker.color = 'rgba(79, 139, 249, 0.15)'
            
            st.plotly_chart(style_plot(fig_p), use_container_width=True)
            
            # Forecast Stats
            pred_price = forecast.iloc[-1]['yhat']
            trend = "Bullish 🟢" if pred_price > current_price else "Bearish 🔴"
            st.info(f"Target Price ({forecast_days} days): **${pred_price:,.2f}** | Trend: **{trend}**")

    # TAB 2: BACKTEST
    with tab_backtest:
        st.markdown("##### 🕵️‍♂️ Reality Check")
        st.caption(f"We hid the last {forecast_days} days of data to see if the AI could predict them.")

        train_len = len(df_train) - forecast_days
        if train_len < 30:
            st.error("Not enough data to run backtest. Increase training history.")
        else:
            train_set = df_train.iloc[:train_len]
            test_set = df_train.iloc[train_len:]
            
            m_bt = Prophet(changepoint_prior_scale=changepoint_scale, seasonality_mode=seasonality_mode)
            m_bt.fit(train_set)
            future_bt = m_bt.make_future_dataframe(periods=forecast_days)
            forecast_bt = m_bt.predict(future_bt)
            
            # Metrics
            combined = pd.merge(test_set, forecast_bt[['ds', 'yhat']], on='ds')
            combined['AbsError'] = (combined['y'] - combined['yhat']).abs()
            mae = combined['AbsError'].mean()
            mape = (combined['AbsError'] / combined['y']).mean() * 100
            
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("MAE (Avg Error)", f"${mae:,.2f}")
            col_b2.metric("MAPE (Error %)", f"{mape:.2f}%")
            
            if mape < 5: col_b3.success("✅ Excellent Accuracy")
            elif mape < 10: col_b3.warning("⚠️ Good Accuracy")
            else: col_b3.error("❌ Poor Accuracy")
            
            # Plot Backtest
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=train_set['ds'], y=train_set['y'], name="Training Data", line=dict(color='#888')))
            fig_bt.add_trace(go.Scatter(x=test_set['ds'], y=test_set['y'], name="Actual Price", line=dict(color='#00FFA3', width=2)))
            fig_bt.add_trace(go.Scatter(x=forecast_bt['ds'], y=forecast_bt['yhat'], name="AI Prediction", line=dict(color='#FF4B4B', dash='dot')))
            
            st.plotly_chart(style_plot(fig_bt), use_container_width=True)

    # TAB 3: CANDLESTICK
    with tab_chart:
        fig_candle = go.Figure(data=[go.Candlestick(
            x=data['Date'],
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name=selected_stock
        )])
        # Add SMAs
        fig_candle.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name="50 SMA", line=dict(color='orange', width=1)))
        fig_candle.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], name="200 SMA", line=dict(color='purple', width=1)))
        
        fig_candle.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(style_plot(fig_candle), use_container_width=True)