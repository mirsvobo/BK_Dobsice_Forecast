import streamlit as st
import pandas as pd
import time
import plotly.express as px
import io
import config

# --- VLASTNÍ MODULY ---
from dataloader import DataLoader
from feature_engineer import FeatureEngineer
from weather_service import WeatherService
from forecast_model import ForecastModel
from diagnostics import get_system_info_markdown
from ui_callback import StreamlitTrainingCallback

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(
    page_title="BK Dobšice AI Forecast",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #EC2934; color: white; font-weight: bold; border-radius: 8px; border: none; }
    .stButton>button:hover { background-color: #C01822; color: white; }
    h1, h2, h3 { color: #502314; }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Burger_King_logo_%281999%29.svg/2024px-Burger_King_logo_%281999%29.svg.png", width=100)
    st.sidebar.title("Nastavení")

    # Zde uživatel mění délku predikce
    forecast_days = st.sidebar.slider("Délka predikce (dny)", 7, 90, 31)

    with st.sidebar.expander("🖥️ Diagnostika", expanded=False):
        st.markdown(get_system_info_markdown())

    force_retrain = st.sidebar.checkbox("Vynutit přetrénování", value=False) # Default False, model si to checkne sám
    run_btn = st.sidebar.button("🚀 SPUSTIT PREDIKCI")

    st.title("🍔 Burger King Dobšice - AI Forecast 3.3")

    target_date = config.TODAY + pd.Timedelta(days=forecast_days)
    st.markdown(f"**Cíl predikce:** {config.TODAY.date()} ➝ {target_date.date()} ({forecast_days} dní)")

    if run_btn:
        run_pipeline(forecast_days, force_retrain)

def run_pipeline(horizon_days, force_retrain):
    start_time = time.time()
    config.FORECAST_HORIZON_DAYS = horizon_days
    pred_end = config.TODAY + pd.Timedelta(days=horizon_days)

    # 1. DATA
    with st.status("Načítám data...", expanded=True) as status:
        loader = DataLoader()
        sales_df, guests_df, lat, lon = loader.load_data()

        if sales_df.empty:
            st.error("Chyba: Žádná data.")
            return

        ws = WeatherService()
        weather_df = ws.get_weather_data(lat, lon, config.TRAIN_START_DATE, pred_end)

        fe = FeatureEngineer()
        sales_df_aug = fe.transform(sales_df, weather_df)
        guests_df_aug = fe.transform(guests_df, weather_df)

        # Future DF (stačí kostra s počasím, IDs si řeší model)
        future_dates = pd.date_range(config.FORECAST_START, pred_end, freq='D')
        future_base = pd.DataFrame({'ds': future_dates})
        future_df_aug = fe.transform(future_base, weather_df)

        S, tags = DataLoader.get_hierarchy_matrix(sales_df)
        status.update(label="Data připravena", state="complete", expanded=False)

    # 2. MODEL
    st.subheader("🧠 Trénink a Predikce")

    col_viz1, col_viz2 = st.columns(2)
    with col_viz1:
        st.markdown("##### Vývoj chyby: Tržby")
        container_sales = st.container()
    with col_viz2:
        st.markdown("##### Vývoj chyby: Hosté")
        container_guests = st.container()

    model = ForecastModel()
    model_loaded = False

    # Zde je změna: load_model nyní přijímá 'required_horizon'
    if not force_retrain:
        model_loaded = model.load_model(config.MODEL_CHECKPOINT_DIR, required_horizon=horizon_days)

    if not model_loaded:
        if force_retrain:
            st.info("Spouštím trénink (vynuceno)...")
        else:
            st.warning(f"Model pro {horizon_days} dní nenalezen, spouštím nový trénink...")

        est_epochs = 2500

        cb_sales = StreamlitTrainingCallback(container_sales, total_epochs=est_epochs, model_name="Tržby (Sales)")
        cb_guests = StreamlitTrainingCallback(container_guests, total_epochs=est_epochs, model_name="Hosté (Guests)")

        # Zde předáváme horizon do tréninku
        model.train(
            sales_df=sales_df_aug,
            guests_df=guests_df_aug,
            horizon=horizon_days,  # <--- DŮLEŽITÉ
            callbacks_sales=[cb_sales],
            callbacks_guests=[cb_guests]
        )
        model.save_model(config.MODEL_CHECKPOINT_DIR)
    else:
        st.success("✅ Model úspěšně načten z disku.")

    # 3. PREDIKCE
    with st.spinner(f"Generuji předpověď na {horizon_days} dní..."):
        preds_sales, preds_guests = model.predict(future_df_aug, S, tags)

    # 4. VÝSLEDKY & EXPORT
    st.divider()

    # Filtrace výsledků (pro jistotu, kdyby model vrátil víc)
    output_sales = preds_sales[['ds', 'unique_id', 'Forecast_Value']].rename(columns={'Forecast_Value': 'Sales'})
    output_guests = preds_guests[['ds', 'unique_id', 'Forecast_Value']].rename(columns={'Forecast_Value': 'Guests'})

    final_df = pd.merge(output_sales, output_guests, on=['ds', 'unique_id'], how='outer')

    # Oříznutí podle požadovaného data (pro případ, že model umí víc)
    final_df = final_df[final_df['ds'] <= pred_end]

    # Grafy Total
    total = final_df[final_df['unique_id'] == 'Total']
    if not total.empty:
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(total, x='ds', y='Sales', title='Tržby Total', color_discrete_sequence=['#EC2934']), use_container_width=True)
        c2.plotly_chart(px.line(total, x='ds', y='Guests', title='Hosté Total', color_discrete_sequence=['#003366']), use_container_width=True)

    # Export (Wide format)
    try:
        sales_wide = final_df.pivot_table(index='ds', columns='unique_id', values='Sales', aggfunc='sum')
        sales_wide.columns = [f"{col}_Sales" for col in sales_wide.columns]

        guests_wide = final_df.pivot_table(index='ds', columns='unique_id', values='Guests', aggfunc='sum')
        guests_wide.columns = [f"{col}_Guests" for col in guests_wide.columns]

        wide_df = sales_wide.join(guests_wide)
        wide_df.reset_index(inplace=True)
        wide_df['ds'] = wide_df['ds'].dt.date
    except Exception as e:
        st.error(f"Chyba při přípravě exportu: {e}")
        wide_df = final_df

    st.subheader("📥 Export Dat")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        wide_df.to_excel(writer, index=False, sheet_name="Prehled_po_Kanalech")
        final_df.to_excel(writer, index=False, sheet_name="Raw_Data")

    st.download_button(
        "Stáhnout Detailní Excel",
        buffer.getvalue(),
        f"Forecast_BK_{config.TODAY.date()}_h{horizon_days}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success(f"Hotovo za {time.time() - start_time:.1f} s")

if __name__ == "__main__":
    main()