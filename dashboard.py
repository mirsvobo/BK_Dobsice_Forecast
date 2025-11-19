import os
import sys
import streamlit as st

# --- 1. ENVIRONMENT & CONFIG (Musí být nahoře) ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# --- 2. FUNKCE PRO NAČTENÍ KNIHOVEN (Lazy Loading) ---
# Tím zabráníme tomu, aby se PyTorch/Lightning načítal v procesech, kde to není potřeba,
# a hlavně to vyřeší kruhové závislosti při 'spawn' na Windows.
def get_engine_modules():
    from dataloader import DataLoader
    from feature_engineer import FeatureEngineer
    from weather_service import WeatherService
    from forecast_model import ForecastModel
    from optimizer import ModelOptimizer
    import config
    return DataLoader, FeatureEngineer, WeatherService, ForecastModel, ModelOptimizer, config

import pandas as pd
import plotly.graph_objects as go

# --- 3. HLAVNÍ APLIKACE ---
def main():
    # Page Config musí být první Streamlit příkaz
    st.set_page_config(
        page_title="BK Dobšice Forecast AI",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Styling
    st.markdown("""
        <style>
        h1 { color: #D62300; }
        .stButton>button { background-color: #D62300; color: white; font-weight: bold; border-radius: 8px; }
        .stProgress .st-bo { background-color: #D62300; }
        </style>
        """, unsafe_allow_html=True)

    # Načtení modulů až zde
    DataLoader, FeatureEngineer, WeatherService, ForecastModel, ModelOptimizer, config = get_engine_modules()

    st.title("BK Dobšice: AI Forecast 2.0")
    st.caption(f"Engine: NeuralForecast (TFT) | Hardware: NVIDIA RTX 5070 | Režim: Windows Process Isolation")

    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Nastavení")
    forecast_start_date = st.sidebar.date_input("Start Predikce", value=pd.to_datetime(config.FORECAST_START).date())
    forecast_days = st.sidebar.slider("Horizont (dny)", 1, 31, 30)
    use_optimization = st.sidebar.checkbox("Zapnout Optunu (Auto-Tuning)", value=False, help="Bude trvat déle.")
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
        last_date = df_hist['ds'].max()
        start_view = last_date - pd.Timedelta(days=30)

        hist = df_hist[(df_hist['unique_id'] == unique_id) & (df_hist['ds'] >= start_view)]
        pred = df_pred[df_pred['unique_id'] == unique_id]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], mode='lines', name='Historie', line=dict(color='black', width=1)))
        fig.add_trace(go.Scatter(x=pred['ds'], y=pred['Forecast_Value'], mode='lines', name='AI Predikce', line=dict(color='#D62300', width=3)))

        if 'y_pred_low' in pred.columns and 'y_pred_high' in pred.columns:
            fig.add_trace(go.Scatter(x=pred['ds'], y=pred['y_pred_high'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=pred['ds'], y=pred['y_pred_low'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(214, 35, 0, 0.2)', name='Interval 80%', hoverinfo='skip'))

        fig.update_layout(title=f"Prognóza: {unique_id}", height=500, template="plotly_white", xaxis_title="Datum", yaxis_title="Hodnota")
        return fig

    # --- LOGIC ---
    with st.spinner("Načítám data..."):
        sales_df, guests_df, lat, lon = load_data_cached()

    if sales_df.empty:
        st.error("Nepodařilo se načíst data.")
        st.stop()

    S, tags = DataLoader.get_hierarchy_matrix(sales_df)
    unique_ids = list(sales_df['unique_id'].unique())

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Záznamy", f"{len(sales_df):,}")
    c2.metric("Start Dat", sales_df['ds'].min().strftime('%d.%m.%Y'))
    c3.metric("Konec Dat", sales_df['ds'].max().strftime('%d.%m.%Y'))
    c4.metric("Kanály", len(unique_ids))

    st.divider()
    st.subheader("1. Příprava Dat")

    if 'fe_done' not in st.session_state:
        st.info("Klikni pro přípravu dat.")

    if st.button("🔄 Spustit Data Pipeline"):
        with st.spinner("Stahuji počasí a generuji features..."):
            train_end_dt = pd.to_datetime(forecast_start_date)
            forecast_end_dt = train_end_dt + pd.Timedelta(days=forecast_days)
            weather_df = get_weather_cached(lat, lon, config.TRAIN_START_DATE, str(forecast_end_dt))

            fe = FeatureEngineer()
            sales_aug = fe.transform(sales_df, weather_df)
            guests_aug = fe.transform(guests_df, weather_df)

            st.session_state['sales_aug'] = sales_aug
            st.session_state['guests_aug'] = guests_aug
            st.session_state['weather_df'] = weather_df
            st.session_state['fe'] = fe
            st.session_state['fe_done'] = True
            st.success("Data připravena!")

    if st.session_state.get('fe_done'):
        st.divider()
        st.subheader("2. AI Model (TFT)")

        if st.button("🚀 Spustit Predikci", type="primary"):
            status_container = st.status("Startuji výpočet na RTX 5070...", expanded=True)
            try:
                train_cutoff = pd.to_datetime(forecast_start_date)
                train_sales = st.session_state['sales_aug'][st.session_state['sales_aug']['ds'] < train_cutoff]
                train_guests = st.session_state['guests_aug'][st.session_state['guests_aug']['ds'] < train_cutoff]

                model = ForecastModel()
                model_loaded = False

                if not force_retrain:
                    status_container.write("Hledám uložený model...")
                    model_loaded = model.load_model(config.MODEL_CHECKPOINT_DIR)

                if not model_loaded:
                    status_container.write("Trénuji nový model (může trvat několik minut)...")
                    best_params = None
                    if use_optimization:
                        status_container.write("Běží Optuna optimalizace...")
                        optimizer = ModelOptimizer(train_sales, horizon=24*7, n_trials=10)
                        best_params = optimizer.optimize()

                    model = ForecastModel(best_params=best_params)
                    model.train(train_sales, train_guests)
                    model.save_model(config.MODEL_CHECKPOINT_DIR)
                    status_container.write("Model uložen.")

                status_container.write("Generuji předpověď...")
                dates = pd.date_range(start=forecast_start_date, periods=forecast_days*24, freq='h')
                future_df = pd.concat([pd.DataFrame({'ds': dates, 'unique_id': uid}) for uid in unique_ids])

                fe = st.session_state['fe']
                future_aug = fe.transform(future_df, st.session_state['weather_df'])
                p_sales, p_guests = model.predict(future_aug, S, tags)

                st.session_state['preds_sales'] = p_sales
                st.session_state['preds_guests'] = p_guests
                status_container.update(label="Hotovo! ✅", state="complete", expanded=False)

            except Exception as e:
                status_container.update(label="Chyba!", state="error")
                st.error(f"Error: {e}")
                st.exception(e)

    if 'preds_sales' in st.session_state:
        st.divider()
        st.subheader("3. Výsledky")
        sel_id = st.selectbox("Kanál:", unique_ids)
        st.plotly_chart(plot_interactive(sales_df, st.session_state['preds_sales'], sel_id), use_container_width=True)

        if 'preds_sales' in st.session_state:
            st.divider()
        st.subheader("3. Výsledky")
        sel_id = st.selectbox("Kanál:", unique_ids)
        st.plotly_chart(plot_interactive(sales_df, st.session_state['preds_sales'], sel_id), use_container_width=True)

        # --- 4. EXPORT (TOTO TAM CHYBĚLO) ---
        st.subheader("4. Export Dat")
        c1, c2 = st.columns(2)

        # A) Export do Excelu (Vše v jednom)
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            st.session_state['preds_sales'].to_excel(writer, sheet_name='Sales', index=False)
            st.session_state['preds_guests'].to_excel(writer, sheet_name='Guests', index=False)

        c1.download_button(
            label="📥 Stáhnout Excel (.xlsx)",
            data=buffer.getvalue(),
            file_name="BK_Forecast_Final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # B) Export do CSV (Záloha)
        csv_sales = st.session_state['preds_sales'].to_csv(index=False).encode('utf-8')
        c2.download_button(
            label="📥 Stáhnout CSV (jen Sales)",
            data=csv_sales,
            file_name="forecast_sales.csv",
            mime="text/csv"
        )

# --- 4. ENTRY POINT (TOHLE JE KLÍČ K OPRAVĚ) ---
if __name__ == "__main__":
    main()