import os
import sys
import streamlit as st
import io
import numpy as np # Přidán numpy pro ošetření dělení

# --- 1. ENVIRONMENT & CONFIG ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# --- 2. MODULES ---
def get_engine_modules():
    from dataloader import DataLoader
    from feature_engineer import FeatureEngineer
    from weather_service import WeatherService
    from forecast_model import ForecastModel
    from optimizer import ModelOptimizer
    import config
    return DataLoader, FeatureEngineer, WeatherService, ForecastModel, ModelOptimizer, config

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

@st.cache_data
def get_hourly_distribution_profile():
    import config
    try:
        if config.DATA_FILE.lower().endswith('.csv'):
            df = pd.read_csv(config.DATA_FILE)
        else:
            df = pd.read_excel(config.DATA_FILE, sheet_name=config.SHEET_NAME)
    except Exception as e:
        st.error(f"Chyba profilu: {e}")
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        config.DATE_COLUMN: 'date',
        config.SALES_COLUMN: 'y',
        config.CHANNEL_COLUMN: 'unique_id'
    }
    time_col = next((c for c in df.columns if 'Time' in c or 'Closing' in c), None)
    if time_col: rename_map[time_col] = 'time'
    df = df.rename(columns=rename_map)

    if 'time' not in df.columns:
        df['hour'] = 12
    else:
        df['hour'] = pd.to_datetime(df['time'].astype(str), errors='coerce').dt.hour.fillna(12).astype(int)

    df['ds'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['ds'])
    df['dayofweek'] = df['ds'].dt.dayofweek
    if df['y'].dtype == object:
        df['y'] = df['y'].astype(str).str.replace(',', '.')
    df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0)

    profile = df.groupby(['unique_id', 'dayofweek', 'hour'])['y'].mean().reset_index()
    daily_sum = profile.groupby(['unique_id', 'dayofweek'])['y'].transform('sum')
    profile['share'] = profile['y'] / daily_sum
    profile = profile.fillna(0)
    return profile[['unique_id', 'dayofweek', 'hour', 'share']]

import pandas as pd
import plotly.graph_objects as go

# --- POMOCNÁ FUNKCE: DISTRIBUCE TOTALU DO KANÁLŮ ---
def reconcile_components(df_long):
    """
    Vezme Total a poměrově ho rozdělí mezi ostatní kanály.
    Zajistí: Sum(Channels) == Total
    """
    # Pivot na Wide (řádky=Datum, sloupce=Kanály)
    df_wide = df_long.pivot_table(
        index='ds', columns='unique_id', values='Forecast_Value', aggfunc='sum'
    ).reset_index()

    if 'Total' not in df_wide.columns:
        return df_long # Pokud chybí Total, vracíme původní

    # Seznam kanálů (vše kromě ds a Total)
    channels = [c for c in df_wide.columns if c not in ['ds', 'Total']]

    if not channels:
        return df_long

    # 1. Součet komponent (jak to vidí model jednotlivě)
    current_sum = df_wide[channels].sum(axis=1)
    target_total = df_wide['Total']

    # 2. Výpočet poměru (Target / Current)
    # Ošetření dělení nulou
    ratio = target_total / current_sum
    ratio = ratio.fillna(1.0) # Kde je suma 0, necháme to být (0 * 1 = 0)
    # Pokud je suma 0 ale total > 0, nelze rozdělit -> ratio bude inf.
    # V tom případě nahradíme 0 (kanály zůstanou 0) nebo 1.
    ratio = ratio.replace([np.inf, -np.inf], 0.0)

    # 3. Přepočet kanálů
    mask = current_sum != 0
    for c in channels:
        # Původní hodnota * Ratio
        df_wide.loc[mask, c] = df_wide.loc[mask, c] * ratio[mask]

    # 4. Melt zpátky na Long format (pro grafy a další zpracování)
    df_final = df_wide.melt(id_vars=['ds'], value_name='Forecast_Value', var_name='unique_id')
    return df_final

# --- 3. HLAVNÍ APLIKACE ---
def main():
    st.set_page_config(page_title="BK Forecast AI", layout="wide")
    st.markdown("""<style>h1 { color: #D62300; }.stButton>button { background-color: #D62300; color: white; font-weight: bold; border-radius: 8px; }</style>""", unsafe_allow_html=True)

    DataLoader, FeatureEngineer, WeatherService, ForecastModel, ModelOptimizer, config = get_engine_modules()

    st.title("BK Dobšice: AI Forecast (Daily -> Hourly)")
    st.caption(f"Engine: NeuralForecast (TFT) | Režim: Denní model (Auto-Reconcile)")

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
    use_optimization = st.sidebar.checkbox("Zapnout Optunu", value=False)
    force_retrain = st.sidebar.checkbox("Vynutit přetrénování", value=False)

    # KPI
    S, tags = DataLoader.get_hierarchy_matrix(sales_df)
    unique_ids = list(sales_df['unique_id'].unique())

    c1, c2, c3 = st.columns(3)
    c1.metric("Historie do", last_hist_date.strftime('%d.%m.%Y'))
    c2.metric("Kanály", len(unique_ids))
    c3.metric("Horizont", f"{config.TFT_PARAMS['h']} dní")

    st.divider()

    # 1. FEATURES
    if 'fe_done' not in st.session_state:
        st.info("Klikni pro přípravu dat.")

    if st.button("🔄 Spustit Data Pipeline"):
        with st.spinner("Stahuji počasí..."):
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
        if st.button("🚀 Spustit Predikci (TFT)", type="primary"):
            status = st.status("Pracuji...", expanded=True)
            try:
                train_cutoff = pd.to_datetime(forecast_start_date)
                if train_cutoff > next_day:
                    train_cutoff = next_day

                train_sales = st.session_state['sales_aug'][st.session_state['sales_aug']['ds'] < train_cutoff]
                train_guests = st.session_state['guests_aug'][st.session_state['guests_aug']['ds'] < train_cutoff]

                model = ForecastModel()
                model_loaded = False

                if not force_retrain:
                    status.write("Hledám model...")
                    model_loaded = model.load_model(config.MODEL_CHECKPOINT_DIR)

                if not model_loaded:
                    status.write("Trénuji model...")
                    best_params = None
                    if use_optimization:
                        optimizer = ModelOptimizer(train_sales, horizon=config.TFT_PARAMS['h'], n_trials=10)
                        best_params = optimizer.optimize()

                    model = ForecastModel(best_params=best_params)
                    model.train(train_sales, train_guests)
                    model.save_model(config.MODEL_CHECKPOINT_DIR)
                    status.write("Uloženo.")

                status.write("Generuji předpověď...")
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
                status.update(label="Hotovo! ✅", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Chyba!", state="error")
                st.error(f"Chyba: {e}")
                st.exception(e)

    # 3. VÝSLEDKY & EXPORT
    if 'preds_sales' in st.session_state:
        st.divider()
        st.subheader("3. Výsledky")

        # --- APLIKACE REKONCILIACE (Total -> Channels) ---
        # Upravíme data v paměti pro zobrazení i export
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
            fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], mode='lines', name='Historie', line=dict(color='black', width=1)))
            fig.add_trace(go.Scatter(x=pred['ds'], y=pred['Forecast_Value'], mode='lines+markers', name='AI Predikce', line=dict(color='#D62300', width=3)))
            fig.update_layout(title=f"Prognóza: {unique_id}", height=500, template="plotly_white")
            return fig

        st.plotly_chart(plot_interactive(hist_sales, sales_viz, sel_id), use_container_width=True)

        st.subheader("4. Export Dat")
        col_d1, col_d2 = st.columns(2)

        # --- A) DENNÍ EXPORT (Wide) ---
        with col_d1:
            st.markdown("**Denní (Wide)**")

            sales_pivot = sales_viz.pivot_table(
                index='ds', columns='unique_id', values='Forecast_Value', aggfunc='sum'
            ).reset_index()

            guests_pivot = guests_viz.pivot_table(
                index='ds', columns='unique_id', values='Forecast_Value', aggfunc='sum'
            ).reset_index()

            sales_pivot['ds'] = sales_pivot['ds'].dt.date
            guests_pivot['ds'] = guests_pivot['ds'].dt.date

            buffer_daily = io.BytesIO()
            with pd.ExcelWriter(buffer_daily, engine='openpyxl') as writer:
                sales_pivot.to_excel(writer, sheet_name='Sales_Daily', index=False)
                guests_pivot.to_excel(writer, sheet_name='Guests_Daily', index=False)
            st.download_button("📥 Stáhnout Denní Excel", buffer_daily.getvalue(), "Forecast_Daily.xlsx")

        # --- B) HODINOVÝ EXPORT (Wide) ---
        with col_d2:
            st.markdown("**Hodinový (Wide)**")
            if st.button("🔢 Přepočítat na hodiny"):
                with st.spinner("Počítám..."):
                    hourly_profile = get_hourly_distribution_profile()
                    if hourly_profile.empty:
                        st.error("Chybí profil.")
                    else:
                        # Použijeme už opravená (reconciled) data
                        daily_preds = sales_viz.copy()
                        daily_preds['dayofweek'] = daily_preds['ds'].dt.dayofweek

                        merged = pd.merge(daily_preds, hourly_profile, on=['unique_id', 'dayofweek'], how='left')
                        merged['hour'] = merged['hour'].fillna(12).astype(int)
                        merged['share'] = merged['share'].fillna(0)
                        merged['Forecast_CZK'] = merged['Forecast_Value'] * merged['share']
                        merged['Final_Date_Time'] = merged.apply(lambda x: x['ds'] + pd.Timedelta(hours=x['hour']), axis=1)

                        # Pivot na Wide
                        export_pivot = merged.pivot_table(
                            index='Final_Date_Time', columns='unique_id', values='Forecast_CZK', aggfunc='sum'
                        ).reset_index()
                        export_pivot = export_pivot.sort_values('Final_Date_Time')

                        # Total v hodinovém exportu vznikne automaticky součtem (protože zdrojová data sedí)
                        # Ale pro jistotu ho můžeme přepočítat
                        channels_h = [c for c in export_pivot.columns if c not in ['Final_Date_Time', 'Total']]
                        if channels_h:
                            export_pivot['Total'] = export_pivot[channels_h].sum(axis=1)

                        buffer_hourly = io.BytesIO()
                        with pd.ExcelWriter(buffer_hourly, engine='openpyxl') as writer:
                            export_pivot.to_excel(writer, sheet_name='Hourly_Sales', index=False)
                        st.session_state['hourly_buffer'] = buffer_hourly
                        st.success("Hotovo!")

            if 'hourly_buffer' in st.session_state:
                st.download_button("📥 Stáhnout Hodinový Excel", st.session_state['hourly_buffer'].getvalue(), "Forecast_Hourly.xlsx")

if __name__ == "__main__":
    main()