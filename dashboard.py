import os
import sys
import streamlit as st
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import subprocess
import json
import time
from itertools import product

# --- 1. ENVIRONMENT & CONFIG ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
torch.set_float32_matmul_precision('medium')

# --- 2. MODULES ---
def get_engine_modules():
    from dataloader import DataLoader
    from feature_engineer import FeatureEngineer
    from weather_service import WeatherService
    from forecast_model import ForecastModel
    from optimizer import ModelOptimizer
    import config
    from ui_layouts import PLTrainingUI, OptunaStreamlitCallback
    return DataLoader, FeatureEngineer, WeatherService, ForecastModel, ModelOptimizer, config, PLTrainingUI, OptunaStreamlitCallback

# --- CACHED DATA LOADERS ---
@st.cache_data
def load_data_cached():
    from dataloader import DataLoader
    loader = DataLoader()
    return loader.load_data()

@st.cache_data
def get_weather_cached(lat, lon, start, end):
    from weather_service import WeatherService
    ws = WeatherService()
    return ws.get_weather_data(lat, lon, str(start), str(end))

# --- POMOCNÉ FUNKCE ---
def reconcile_components(df_long):
    df_wide = df_long.pivot_table(index='ds', columns='unique_id', values='Forecast_Value', aggfunc='sum', fill_value=0).reset_index()
    if 'Total' not in df_wide.columns: return df_long
    channels = [c for c in df_wide.columns if c not in ['ds', 'Total']]
    if not channels: return df_long
    current_sum = df_wide[channels].sum(axis=1)
    target_total = df_wide['Total']
    ratio = target_total / current_sum
    ratio = ratio.fillna(1.0).replace([np.inf, -np.inf], 0.0)
    mask = current_sum != 0
    for c in channels:
        df_wide.loc[mask, c] = df_wide.loc[mask, c] * ratio[mask]
    return df_wide.melt(id_vars=['ds'], value_name='Forecast_Value', var_name='unique_id')

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {int(s)}s"

# --- 3. HLAVNÍ APLIKACE ---
def main():
    st.set_page_config(page_title="BK Forecast AI", layout="wide")
    st.markdown("""
        <style>
        h1 { color: #D62300; }
        .stButton>button { background-color: #D62300; color: white; font-weight: bold; border-radius: 8px; }
        .big-font { font-size:24px !important; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)

    DataLoader, FeatureEngineer, WeatherService, ForecastModel, ModelOptimizer, config, PLTrainingUI, OptunaStreamlitCallback = get_engine_modules()

    st.title("BK Dobšice: AI Forecast (Pro Dashboard)")

    hw_info = "CPU Mode"
    if torch.cuda.is_available():
        hw_info = f"GPU: {torch.cuda.get_device_name(0)}"
    st.caption(f"Engine: NeuralForecast (TFT) | {hw_info}")

    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Nastavení")
    with st.spinner("Analyzuji data..."):
        sales_df, _, _, _ = load_data_cached()
        last_hist_date = sales_df['ds'].max()
        next_day = last_hist_date + pd.Timedelta(days=1)

    forecast_start_date = st.sidebar.date_input("Start Predikce", value=next_day.date())
    st.sidebar.caption(f"Horizont: {config.TFT_PARAMS['h']} dní")
    st.sidebar.divider()

    use_optuna = st.sidebar.checkbox("Zapnout Optunu (Auto-Tuning)", value=False)
    optuna_trials = 10
    if use_optuna:
        optuna_trials = st.sidebar.slider("Počet pokusů Optuny", 5, 50, 10)

    force_retrain = st.sidebar.checkbox("Vynutit přetrénování", value=False)

    unique_ids = list(sales_df['unique_id'].unique())
    S, tags = DataLoader.get_hierarchy_matrix(sales_df)
    st.divider()

    # 1. FEATURES
    if 'fe_done' not in st.session_state:
        st.info("Klikni pro přípravu dat.")

    if st.button("🔄 Spustit Data Pipeline"):
        with st.spinner("Stahuji počasí a připravuji features..."):
            sales_df, guests_df, lat, lon = load_data_cached()
            train_end_dt = pd.to_datetime(forecast_start_date)
            forecast_end_dt = train_end_dt + pd.Timedelta(days=config.TFT_PARAMS['h'] + 10)
            weather_df = get_weather_cached(lat, lon, config.TRAIN_START_DATE, str(forecast_end_dt))

            fe = FeatureEngineer()
            sales_aug = fe.transform(sales_df, weather_df)
            guests_aug = fe.transform(guests_df, weather_df)

            st.session_state['sales_aug'] = sales_aug
            st.session_state['guests_aug'] = guests_aug
            st.session_state['weather_df'] = weather_df
            st.session_state['fe'] = fe
            st.session_state['fe_done'] = True
            st.success("Data připravena.")

    # 2. MODEL
    if st.session_state.get('fe_done'):
        st.divider()
        if st.button("🚀 Spustit Trénink a Predikci", type="primary"):
            viz_container = st.container()

            try:
                train_cutoff = pd.to_datetime(forecast_start_date)
                if train_cutoff > next_day: train_cutoff = next_day

                train_sales = st.session_state['sales_aug'][st.session_state['sales_aug']['ds'] < train_cutoff]
                train_guests = st.session_state['guests_aug'][st.session_state['guests_aug']['ds'] < train_cutoff]

                # --- VÝPOČET IQR PRO ODHAD REÁLNÉ CHYBY (CZK) ---
                # RobustScaler dělí data podle IQR (Q75 - Q25).
                # Abychom z Loss (škálované) dostali Kč, musíme ji tímto číslem vynásobit.
                sales_values = train_sales[train_sales['unique_id'] == 'Total']['y']
                if not sales_values.empty:
                    q75, q25 = np.percentile(sales_values, [75 ,25])
                    iqr_scale = q75 - q25
                else:
                    iqr_scale = 10000.0 # Fallback
                # ------------------------------------------------

                model = ForecastModel()
                model_loaded = False

                if not force_retrain:
                    with viz_container: st.info("🔎 Hledám uložený model...")
                    model_loaded = model.load_model(config.MODEL_CHECKPOINT_DIR)

                if not model_loaded:
                    best_params = {}
                    if use_optuna:
                        optuna_cont = viz_container.container()
                        with optuna_cont:
                            optuna_cb = OptunaStreamlitCallback(optuna_cont, optuna_trials)
                            optimizer = ModelOptimizer(train_sales, horizon=config.TFT_PARAMS['h'], n_trials=optuna_trials)
                            best_params = optimizer.optimize(streamlit_callback=optuna_cb)
                            st.success(f"✅ Optuna dokončena! Nejlepší parametry: {best_params}")
                            st.divider()

                    # --- B) FINÁLNÍ TRÉNINK (SUBPROCESS) ---
                    training_cont = viz_container.container()
                    with training_cont:
                        st.markdown("## 🏎️ Produkční Trénink")

                        # --- ROZLOŽENÍ DASHBOARDU ---
                        status_text = st.empty()
                        prog_bar = st.progress(0)

                        # 4 Sloupce pro metriky
                        m1, m2, m3, m4 = st.columns(4)
                        metric_step = m1.empty()
                        metric_real_loss = m2.empty() # Tady budou peníze
                        metric_speed = m3.empty()
                        metric_eta = m4.empty()

                        # Grafy
                        col_g1, col_g2 = st.columns([2, 1])
                        chart_loss = col_g1.empty()
                        chart_speed = col_g2.empty()

                        loss_history = []
                        step_history = []
                        speed_history = []
                        training_success = False

                        # 1. Příprava souborů
                        temp_sales_file = "temp_train_sales.csv"
                        temp_guests_file = "temp_train_guests.csv"
                        temp_params_file = "temp_params.json"

                        train_sales.to_csv(temp_sales_file, index=False)
                        train_guests.to_csv(temp_guests_file, index=False)
                        with open(temp_params_file, 'w') as f: json.dump(best_params, f)

                        # 2. Start procesu
                        cmd = [sys.executable, "train_runner.py", temp_sales_file, temp_guests_file, temp_params_file]
                        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

                        # 3. Čtení metrik
                        while True:
                            line = process.stdout.readline()
                            if not line and process.poll() is not None: break

                            if line:
                                try:
                                    msg = json.loads(line.strip())

                                    if msg['type'] == 'status':
                                        status_text.markdown(f"**Status:** {msg['msg']}")

                                    elif msg['type'] == 'metrics':
                                        s = msg['step']; t = msg['total']; l = msg['loss']
                                        speed = msg.get('speed', 0)
                                        eta = msg.get('eta', 0)

                                        # Výpočet reálné chyby v Kč
                                        real_mae_est = l * iqr_scale

                                        # Update Metrik
                                        metric_step.metric("Kroky", f"{s} / {t}")
                                        metric_real_loss.metric("Odhad chyby (MAE)", f"{real_mae_est:,.0f} Kč", help="Přepočteno z Loss pomocí IQR škálování.")
                                        metric_speed.metric("Rychlost", f"{speed:.2f} kroků/s")
                                        metric_eta.metric("Zbývá (ETA)", format_time(eta))

                                        prog_bar.progress(min(s/t, 1.0))

                                        # Grafy
                                        loss_history.append(l)
                                        step_history.append(s)
                                        speed_history.append(speed)

                                        # Main chart: Loss
                                        df_loss = pd.DataFrame({'Training Loss': loss_history}, index=step_history)
                                        chart_loss.line_chart(df_loss, height=250)

                                        # Secondary chart: Speed
                                        # df_speed = pd.DataFrame({'Rychlost': speed_history}, index=step_history)
                                        # chart_speed.line_chart(df_speed, height=250) # Volitelné, pokud chcete

                                    elif msg['type'] == 'done':
                                        status_text.success(msg['msg'])
                                        prog_bar.progress(1.0)
                                        training_success = True
                                    elif msg['type'] == 'error':
                                        st.error(f"Chyba: {msg['msg']}")
                                        training_success = False
                                except json.JSONDecodeError:
                                    pass

                        # Úklid
                        if os.path.exists(temp_sales_file): os.remove(temp_sales_file)
                        if os.path.exists(temp_guests_file): os.remove(temp_guests_file)
                        if os.path.exists(temp_params_file): os.remove(temp_params_file)

                        if not training_success:
                            st.error("❌ Trénink selhal.")
                            st.stop()

                        st.success("✅ Model úspěšně natrénován.")
                        model = ForecastModel()
                        model.load_model(config.MODEL_CHECKPOINT_DIR)

                # 3. PREDIKCE
                with viz_container: st.info("🔮 Generuji předpověď do budoucnosti...")

                horizon = config.TFT_PARAMS['h']
                future_df = model.nf_sales.make_future_dataframe(h=horizon)
                future_df = future_df.reset_index()
                if 'unique_id' not in future_df.columns and 'index' in future_df.columns:
                    future_df = future_df.rename(columns={'index': 'unique_id'})

                fe = st.session_state['fe']
                future_aug = fe.transform(future_df, st.session_state['weather_df'])

                model_ids = future_df['unique_id'].unique()
                S_pred = pd.DataFrame(np.eye(len(model_ids)), index=model_ids, columns=model_ids)
                tags_pred = {'Country': model_ids, 'Country/State': model_ids}

                p_sales, p_guests = model.predict(future_aug, S_pred, tags_pred)
                st.session_state['preds_sales'] = p_sales
                st.session_state['preds_guests'] = p_guests

                with viz_container:
                    st.success("Hotovo! Výsledky jsou níže.")
                    st.balloons()

            except Exception as e:
                viz_container.error("Nastala chyba!")
                st.exception(e)

    # 3. VÝSLEDKY (Stejné jako předtím)
    if 'preds_sales' in st.session_state:
        st.divider()
        st.subheader("3. Výsledky")
        sales_viz = reconcile_components(st.session_state['preds_sales'])
        sel_id = st.selectbox("Vyber kanál:", unique_ids)
        hist_sales, _, _, _ = load_data_cached()

        last_date = hist_sales['ds'].max()
        hist = hist_sales[(hist_sales['unique_id'] == sel_id) & (hist_sales['ds'] >= last_date - pd.Timedelta(days=60))]
        pred = sales_viz[sales_viz['unique_id'] == sel_id]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], name='Historie', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=pred['ds'], y=pred['Forecast_Value'], name='Predikce', line=dict(color='#D62300', width=3)))
        st.plotly_chart(fig, use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            sales_viz.to_excel(writer, sheet_name='Sales', index=False)
        st.download_button("📥 Stáhnout Excel", buffer.getvalue(), "forecast.xlsx")

if __name__ == "__main__":
    main()