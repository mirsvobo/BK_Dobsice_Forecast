import os
# --- CONFIG PAMĚTI ---
# Nastavíme to, ale VOLAT torch.cuda NEBUDEME
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import gc
# import torch  <-- Tady to sice importujeme, ale nesmíme volat funkce na GPU

# Vlastní moduly
import config
from dataloader import DataLoader
from feature_engineer import FeatureEngineer
from weather_service import WeatherService
from forecast_model import ForecastModel
from optimizer import ModelOptimizer

# --- PAGE CONFIG ---
st.set_page_config(page_title="BK Dobšice Forecast AI", page_icon="🍔", layout="wide")

# Styling
st.markdown("""
    <style>
    h1 { color: #D62300; }
    .stButton>button { background-color: #D62300; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🍔 Burger King Dobšice: AI Forecast")
st.markdown(f"**Engine:** NeuralForecast (TFT) | **Hardware:** RTX 5070 (Process Isolation) 🚀")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Nastavení")
forecast_start_date = st.sidebar.date_input("Start Predikce", value=pd.to_datetime(config.FORECAST_START).date())
forecast_days = st.sidebar.slider("Délka předpovědi (dny)", 1, 31, 30)
use_optimization = st.sidebar.checkbox("Zapnout Optunu (Auto-Tuning)", value=False)
force_retrain = st.sidebar.checkbox("Vynutit přetrénování", value=False)

# --- CACHED FUNCTIONS ---
@st.cache_data
def load_data_cached():
    loader = DataLoader()
    return loader.load_data()

@st.cache_data
def get_weather_cached(lat, lon, start, end):
    ws = WeatherService()
    return ws.get_weather_data(lat, lon, str(start), str(end))

# --- PLOT FUNCTION ---
def plot_interactive(df_hist, df_pred, unique_id):
    hist = df_hist[df_hist['unique_id'] == unique_id]
    pred = df_pred[df_pred['unique_id'] == unique_id]

    last_date = hist['ds'].max()
    start_view = last_date - pd.Timedelta(days=14)
    hist = hist[hist['ds'] >= start_view]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], mode='lines', name='Historie', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=pred['ds'], y=pred['Forecast_Value'], mode='lines', name='Predikce (AI)', line=dict(color='#D62300', width=2)))

    if 'y_pred_low' in pred.columns:
        fig.add_trace(go.Scatter(x=pred['ds'], y=pred['y_pred_high'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=pred['ds'], y=pred['y_pred_low'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(214, 35, 0, 0.2)', name='Interval'))

    fig.update_layout(title=f"Prognóza: {unique_id}", height=500, template="plotly_white")
    return fig

# --- APP LOGIC ---
with st.spinner("Načítám data..."):
    sales_df, guests_df, lat, lon = load_data_cached()

if sales_df.empty:
    st.error("Chyba dat.")
    st.stop()

S, tags = DataLoader.get_hierarchy_matrix(sales_df)
unique_ids = list(sales_df['unique_id'].unique())

# KPI
col1, col2, col3 = st.columns(3)
last_week_sales = sales_df[sales_df['ds'] >= sales_df['ds'].max() - pd.Timedelta(days=7)]['y'].sum()
col1.metric("Záznamy", len(sales_df))
col2.metric("Tržby (7 dní)", f"{last_week_sales:,.0f} Kč")
col3.metric("Kanály", len(unique_ids))

st.write("---")
st.header("1. Příprava Features")

if st.button("🔄 Aktualizovat Features"):
    with st.spinner("Zpracovávám..."):
        end_dt = pd.to_datetime(forecast_start_date) + pd.Timedelta(days=forecast_days)
        weather_df = get_weather_cached(lat, lon, config.TRAIN_START_DATE, str(end_dt))

        fe = FeatureEngineer()
        sales_aug = fe.transform(sales_df, weather_df)
        guests_aug = fe.transform(guests_df, weather_df)

        st.session_state['sales_aug'] = fe.create_lags(sales_aug)
        st.session_state['guests_aug'] = fe.create_lags(guests_aug)
        st.session_state['weather_df'] = weather_df
        st.session_state['fe'] = fe
        st.success("Hotovo.")

if 'sales_aug' in st.session_state:
    st.write("---")
    st.header("2. Trénink Modelu")

    if st.button("🚀 Spustit AI Model"):
        progress_bar = st.progress(0)
        status = st.empty()

        train_cutoff = pd.to_datetime(forecast_start_date)
        train_sales = st.session_state['sales_aug'][st.session_state['sales_aug']['ds'] < train_cutoff]
        train_guests = st.session_state['guests_aug'][st.session_state['guests_aug']['ds'] < train_cutoff]

        model = ForecastModel()
        model_loaded = False

        if not force_retrain:
            status.text("Zkouším načíst uložený model...")
            model_loaded = model.load_model(config.MODEL_CHECKPOINT_DIR)

        if model_loaded:
            status.success("Model načten z disku.")
            st.session_state['model'] = model
            progress_bar.progress(100)
        else:
            status.text("Hledám parametry a trénuji na GPU...")
            best_params = None
            if use_optimization:
                optimizer = ModelOptimizer(train_sales, horizon=24*forecast_days, n_trials=10)
                best_params = optimizer.optimize()

            model = ForecastModel(best_params=best_params)
            model.train(train_sales, train_guests)
            model.save_model(config.MODEL_CHECKPOINT_DIR)
            st.session_state['model'] = model
            status.success("Nový model natrénován!")
            progress_bar.progress(100)

    if 'model' in st.session_state:
        st.write("---")
        st.header("3. Výsledky")

        if st.button("🔮 Predikovat"):
            dates = pd.date_range(start=forecast_start_date, periods=forecast_days*24, freq='H')
            future_df = pd.DataFrame()
            for uid in unique_ids:
                future_df = pd.concat([future_df, pd.DataFrame({'ds': dates, 'unique_id': uid})])

            fe = st.session_state['fe']
            future_aug = fe.transform(future_df, st.session_state['weather_df'])

            model = st.session_state['model']
            p_sales, p_guests = model.predict(future_aug, S, tags)

            st.session_state['preds_sales'] = p_sales
            st.session_state['preds_guests'] = p_guests
            st.success("Predikce hotova.")

        if 'preds_sales' in st.session_state:
            sel_id = st.selectbox("Vyber kanál:", unique_ids)
            st.plotly_chart(plot_interactive(sales_df, st.session_state['preds_sales'], sel_id), use_container_width=True)

            c1, c2 = st.columns(2)
            c1.download_button("Stáhnout Sales (CSV)", st.session_state['preds_sales'].to_csv(index=False), "sales.csv")
            c2.download_button("Stáhnout Guests (CSV)", st.session_state['preds_guests'].to_csv(index=False), "guests.csv")