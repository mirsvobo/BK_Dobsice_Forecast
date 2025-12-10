import os
import sys
import streamlit as st
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

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
    # Import nových UI layoutů
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

# --- POMOCNÁ FUNKCE: DISTRIBUCE TOTALU DO KANÁLŮ ---
def reconcile_components(df_long):
    """
    Vezme predikci pro 'Total' a poměrově ji rozdělí mezi ostatní kanály.
    Zajistí: Sum(Kanály) == Total
    """
    # Pivot na Wide (řádky=Datum, sloupce=Kanály)
    df_wide = df_long.pivot_table(
        index='ds', columns='unique_id', values='Forecast_Value', aggfunc='sum', fill_value=0
    ).reset_index()

    if 'Total' not in df_wide.columns:
        return df_long

    channels = [c for c in df_wide.columns if c not in ['ds', 'Total']]
    if not channels:
        return df_long

    # 1. Součet komponent (jak to vidí model jednotlivě)
    current_sum = df_wide[channels].sum(axis=1)
    target_total = df_wide['Total']

    # 2. Výpočet poměru (Kolikrát musíme kanály zvětšit/zmenšit, aby daly Total)
    ratio = target_total / current_sum
    ratio = ratio.fillna(1.0).replace([np.inf, -np.inf], 0.0)

    # 3. Přepočet kanálů
    mask = current_sum != 0
    for c in channels:
        df_wide.loc[mask, c] = df_wide.loc[mask, c] * ratio[mask]

    # 4. Melt zpátky na Long format (pro grafy)
    return df_wide.melt(id_vars=['ds'], value_name='Forecast_Value', var_name='unique_id')

# --- 3. HLAVNÍ APLIKACE ---
def main():
    st.set_page_config(page_title="BK Forecast AI", layout="wide")
    st.markdown("""<style>h1 { color: #D62300; }.stButton>button { background-color: #D62300; color: white; font-weight: bold; border-radius: 8px; }</style>""", unsafe_allow_html=True)

    DataLoader, FeatureEngineer, WeatherService, ForecastModel, ModelOptimizer, config, PLTrainingUI, OptunaStreamlitCallback = get_engine_modules()

    st.title("BK Dobšice: AI Forecast (RTX 5070 Ed.)")

    # Detekce HW
    hw_info = "CPU Mode"
    if torch.cuda.is_available():
        hw_info = f"GPU: {torch.cuda.get_device_name(0)}"

    st.caption(f"Engine: NeuralForecast (TFT) | {hw_info} | Batch Size: {config.TFT_PARAMS['batch_size']}")

    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Nastavení")

    with st.spinner("Analyzuji data..."):
        sales_df, _, _, _ = load_data_cached()
        last_hist_date = sales_df['ds'].max()
        next_day = last_hist_date + pd.Timedelta(days=1)

    forecast_start_date = st.sidebar.date_input(
        "Start Predikce",
        value=next_day.date(),
        help="Automaticky nastaveno na den po konci dat."
    )

    st.sidebar.caption(f"Horizont: {config.TFT_PARAMS['h']} dní")

    st.sidebar.divider()
    # Optuna nastavení
    use_optuna = st.sidebar.checkbox(
        "Zapnout Optunu (Auto-Tuning)",
        value=False,
        help="Hledá nejlepší parametry. Spustí více tréninků, zobrazí výsledky."
    )

    # Pokud je Optuna zapnutá, ukážeme slider pro počet pokusů
    optuna_trials = 0
    if use_optuna:
        optuna_trials = st.sidebar.slider("Počet pokusů Optuny", min_value=5, max_value=50, value=10)

    force_retrain = st.sidebar.checkbox("Vynutit přetrénování", value=False)

    # KPI
    unique_ids = list(sales_df['unique_id'].unique())
    S, tags = DataLoader.get_hierarchy_matrix(sales_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Historie do", last_hist_date.strftime('%d.%m.%Y'))
    c2.metric("Kanály", len(unique_ids))
    c3.metric("Horizont", f"{config.TFT_PARAMS['h']} dní")

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

            # --- HLAVNÍ KONTEJNER PRO VIZUALIZACI ---
            viz_container = st.container()

            try:
                train_cutoff = pd.to_datetime(forecast_start_date)
                if train_cutoff > next_day:
                    train_cutoff = next_day

                train_sales = st.session_state['sales_aug'][st.session_state['sales_aug']['ds'] < train_cutoff]
                train_guests = st.session_state['guests_aug'][st.session_state['guests_aug']['ds'] < train_cutoff]

                model = ForecastModel()
                model_loaded = False

                if not force_retrain:
                    with viz_container:
                        st.info("🔎 Hledám uložený model...")
                    model_loaded = model.load_model(config.MODEL_CHECKPOINT_DIR)

                if not model_loaded:
                    best_params = None

                    # --- A) OPTUNA VIZUALIZACE ---
                    if use_optuna:
                        optuna_cont = viz_container.container()
                        with optuna_cont:
                            # Inicializace callbacku pro Optunu
                            optuna_cb = OptunaStreamlitCallback(optuna_cont, optuna_trials)

                            optimizer = ModelOptimizer(train_sales, horizon=config.TFT_PARAMS['h'], n_trials=optuna_trials)

                            # Spuštění s callbackem
                            best_params = optimizer.optimize(streamlit_callback=optuna_cb)

                            st.success(f"✅ Optuna dokončena! Nejlepší parametry: {best_params}")
                            st.divider()

                    # --- B) FINÁLNÍ TRÉNINK VIZUALIZACE ---
                    training_cont = viz_container.container()
                    with training_cont:
                        training_cont.markdown("## 🏭 Produkční Trénink Modelu")

                        # Vlastní trénink s UI vizualizací
                        model = ForecastModel(best_params=best_params)
                        model.train(
                            train_sales,
                            train_guests,
                            ui_callback_cls=PLTrainingUI,
                            ui_container=training_cont
                        )

                        model.save_model(config.MODEL_CHECKPOINT_DIR)
                        st.success("✅ Model natrénován a uložen.")

                # 3. PREDIKCE
                with viz_container:
                    st.info("🔮 Generuji předpověď do budoucnosti...")

                horizon = config.TFT_PARAMS['h']
                dates = pd.date_range(start=train_cutoff, periods=horizon, freq='D')

                future_df = pd.DataFrame()
                for uid in unique_ids:
                    tmp = pd.DataFrame({'ds': dates, 'unique_id': uid})
                    future_df = pd.concat([future_df, tmp])

                fe = st.session_state['fe']
                future_aug = fe.transform(future_df, st.session_state['weather_df'])

                p_sales, p_guests = model.predict(future_aug, S, tags)
                st.session_state['preds_sales'] = p_sales
                st.session_state['preds_guests'] = p_guests

                with viz_container:
                    st.success("Hotovo! Výsledky jsou níže.")
                    st.balloons()

            except Exception as e:
                viz_container.error("Nastala chyba!")
                st.error(f"Detaily chyby: {e}")
                st.exception(e)

    # 3. VÝSLEDKY & EXPORT
    if 'preds_sales' in st.session_state:
        st.divider()
        st.subheader("3. Výsledky")

        sales_viz = reconcile_components(st.session_state['preds_sales'])
        guests_viz = reconcile_components(st.session_state['preds_guests'])

        sel_id = st.selectbox("Vyber kanál:", unique_ids)
        hist_sales, _, _, _ = load_data_cached()

        def plot_interactive(df_hist, df_pred, unique_id):
            last_date = df_hist['ds'].max()
            start_view = last_date - pd.Timedelta(days=60)
            hist = df_hist[(df_hist['unique_id'] == unique_id) & (df_hist['ds'] >= start_view)]
            pred = df_pred[df_pred['unique_id'] == unique_id]

            fig = go.Figure()
            # Historie
            fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], mode='lines', name='Historie', line=dict(color='gray', width=1)))
            # Predikce
            fig.add_trace(go.Scatter(x=pred['ds'], y=pred['Forecast_Value'], mode='lines+markers', name='AI Predikce', line=dict(color='#D62300', width=3)))

            # Confidence Interval (stínování)
            if 'y_pred_low' in pred.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([pred['ds'], pred['ds'][::-1]]),
                    y=pd.concat([pred['y_pred_high'], pred['y_pred_low'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(214, 35, 0, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))

            fig.update_layout(title=f"Prognóza: {unique_id}", height=500, template="plotly_white")
            return fig

        st.plotly_chart(plot_interactive(hist_sales, sales_viz, sel_id), use_container_width=True)

        st.subheader("4. Export Dat")
        sales_pivot = sales_viz.pivot_table(index='ds', columns='unique_id', values='Forecast_Value', aggfunc='sum', fill_value=0).reset_index()
        guests_pivot = guests_viz.pivot_table(index='ds', columns='unique_id', values='Forecast_Value', aggfunc='sum', fill_value=0).reset_index()

        cols_s = [c for c in sales_pivot.columns if c not in ['ds', 'Total']]
        if cols_s: sales_pivot['Total'] = sales_pivot[cols_s].sum(axis=1)

        cols_g = [c for c in guests_pivot.columns if c not in ['ds', 'Total']]
        if cols_g: guests_pivot['Total'] = guests_pivot[cols_g].sum(axis=1)

        # Rounding & Int
        num_s = [c for c in sales_pivot.columns if c != 'ds']
        sales_pivot[num_s] = sales_pivot[num_s].round(0).astype(int)
        num_g = [c for c in guests_pivot.columns if c != 'ds']
        guests_pivot[num_g] = guests_pivot[num_g].round(0).astype(int)

        sales_pivot['ds'] = sales_pivot['ds'].dt.date
        guests_pivot['ds'] = guests_pivot['ds'].dt.date

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            sales_pivot.to_excel(writer, sheet_name='Sales_Forecast', index=False)
            guests_pivot.to_excel(writer, sheet_name='Transactions_Forecast', index=False)

        st.download_button(
            label="📥 Stáhnout Predikci (.xlsx)",
            data=buffer.getvalue(),
            file_name="BK_Forecast_Daily.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()