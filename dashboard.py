import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go

# Import našich optimalizovaných modulů
import config
from dataloader import DataLoader
from feature_engineer import FeatureEngineer
from weather_service import WeatherService
from forecast_model import ForecastModel
from diagnostics import get_system_info_markdown

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(
    page_title="BK Dobšice AI Forecast",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #EC2934;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # --- SIDEBAR: OVLÁDÁNÍ ---
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Burger_King_logo_%281999%29.svg/2024px-Burger_King_logo_%281999%29.svg.png", width=100)
    st.sidebar.title("Nastavení Predikce")

    # 1. Slicer na dny
    forecast_days = st.sidebar.slider("Délka predikce (dny)", min_value=7, max_value=90, value=31, step=1)

    # 2. Diagnostika HW (vždy viditelná)
    with st.sidebar.expander("🖥️ Diagnostika Hardware", expanded=True):
        hw_info = get_system_info_markdown()
        st.markdown(hw_info)

    st.sidebar.markdown("---")
    force_retrain = st.sidebar.checkbox("Vynutit přetrénování modelu", value=True, help="Pokud je vypnuto, aplikace se pokusí načíst poslední uložený model.")

    run_btn = st.sidebar.button("🚀 SPUSTIT PREDIKCI")

    # --- HLAVNÍ OBSAH ---
    st.title("🍔 Burger King Dobšice - AI Forecast 3.0")
    st.markdown(f"**Cíl predikce:** {config.TODAY.date()} ➝ {(config.TODAY + pd.Timedelta(days=forecast_days)).date()}")

    if run_btn:
        run_forecasting_pipeline(forecast_days, force_retrain)

def run_forecasting_pipeline(horizon_days, force_retrain):
    start_time = time.time()

    # Aktualizace configu podle slideru
    config.FORECAST_HORIZON_DAYS = horizon_days
    config.TFT_PARAMS['h'] = horizon_days

    # Dynamické nastavení datumu
    prediction_end_date = config.TODAY + pd.Timedelta(days=horizon_days)

    # --- KROK 1: DATA ---
    with st.status("Načítám a zpracovávám data...", expanded=True) as status:
        st.write("📥 Načítám historii prodejů...")
        loader = DataLoader()
        sales_df, guests_df, lat, lon = loader.load_data()

        if sales_df.empty:
            status.update(label="Chyba: Žádná data!", state="error")
            st.error("Nepodařilo se načíst data z Excelu.")
            return

        st.write(f"🌦️ Stahuji počasí (History + Forecast pro {horizon_days} dní)...")
        ws = WeatherService()
        weather_df = ws.get_weather_data(lat, lon, config.TRAIN_START_DATE, prediction_end_date)

        st.write("⚙️ Feature Engineering (Svátky, Eventy, Transforamce)...")
        fe = FeatureEngineer()
        sales_df_aug = fe.transform(sales_df, weather_df)
        guests_df_aug = fe.transform(guests_df, weather_df)

        # Příprava budoucnosti
        future_dates = pd.date_range(start=config.FORECAST_START, end=prediction_end_date, freq='D')
        unique_ids = sales_df['unique_id'].unique()
        future_df = pd.DataFrame([{'ds': d, 'unique_id': uid} for d in future_dates for uid in unique_ids])
        future_df_aug = fe.transform(future_df, weather_df)

        S, tags = DataLoader.get_hierarchy_matrix(sales_df)
        status.update(label="Data připravena!", state="complete", expanded=False)

    # --- KROK 2: AI MODEL ---
    with st.status("Trénuji AI Modely (GPU Accelerated)...", expanded=True) as status:
        model = ForecastModel()

        # Logika Load vs Train
        model_loaded = False
        if not force_retrain:
            st.write("📂 Zkouším načíst uložený model...")
            model_loaded = model.load_model(config.MODEL_CHECKPOINT_DIR)

        if not model_loaded:
            st.write("🧠 Trénuji model: Tržby (Sales)...")
            # Pro Streamlit je lepší progress bar uvnitř modelu řešit přes status update,
            # ale zde necháme běžet backend logy, které se vypisují do konzole, a tady ukážeme spinner.
            model.train(sales_df_aug, guests_df_aug)
            model.save_model(config.MODEL_CHECKPOINT_DIR)
        else:
            st.write("✅ Model úspěšně načten z disku.")

        st.write("🔮 Generuji predikce a rekonsiliuji hierarchii...")
        preds_sales, preds_guests = model.predict(future_df_aug, S, tags)

        status.update(label="Výpočty dokončeny!", state="complete", expanded=False)

    # --- KROK 3: VÝSLEDKY & GRAFY ---
    st.divider()
    st.subheader("📊 Výsledky Predikce")

    # Příprava dat pro zobrazení
    # Spojení Sales a Guests
    output_sales = preds_sales[['ds', 'unique_id', 'Forecast_Value']].rename(columns={'Forecast_Value': 'Sales'})
    output_guests = preds_guests[['ds', 'unique_id', 'Forecast_Value']].rename(columns={'Forecast_Value': 'Guests'})
    final_df = pd.merge(output_sales, output_guests, on=['ds', 'unique_id'], how='outer')

    # 1. Total Graf
    total_df = final_df[final_df['unique_id'] == 'Total']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 💰 Očekávané Tržby (Total)")
        fig_sales = px.line(total_df, x='ds', y='Sales', title='Projekce Tržeb (Total)', markers=True)
        fig_sales.update_traces(line_color='#EC2934')
        st.plotly_chart(fig_sales, use_container_width=True)

        total_sum_sales = total_df['Sales'].sum()
        st.info(f"Celkové predikované tržby: **{total_sum_sales:,.0f} Kč**")

    with col2:
        st.markdown("### 👥 Očekávaní Hosté (Total)")
        fig_guests = px.line(total_df, x='ds', y='Guests', title='Projekce Hostů (Total)', markers=True)
        fig_guests.update_traces(line_color='#003366')
        st.plotly_chart(fig_guests, use_container_width=True)

        total_sum_guests = total_df['Guests'].sum()
        st.info(f"Celkový počet hostů: **{total_sum_guests:,.0f}**")

    # 2. Kanály Graf
    st.markdown("### 🍟 Rozpad po kanálech")
    channels_df = final_df[final_df['unique_id'] != 'Total']
    fig_channels = px.bar(channels_df, x='ds', y='Sales', color='unique_id', title='Denní tržby dle kanálu')
    st.plotly_chart(fig_channels, use_container_width=True)

    # --- KROK 4: EXPORT ---
    st.divider()

    # Uložení do Excelu v paměti pro download button
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    filename = f"BK_Forecast_{timestamp}.xlsx"

    # Excel writer buffer
    import io
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        final_df.to_excel(writer, index=False, sheet_name='Forecast')
        sales_df_aug.to_excel(writer, index=False, sheet_name='Training_Data_Debug')

    st.download_button(
        label="📥 Stáhnout predikci do Excelu",
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    end_time = time.time()
    st.success(f"Hotovo za {end_time - start_time:.1f} sekund.")

if __name__ == "__main__":
    main()