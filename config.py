import os

# --- 1. Konfigurace cest a souborů ---
DATA_FILE = 'data.xlsx'
SHEET_NAME = "OPSBKSalesMixOverview4"
WEATHER_CACHE_DIR = ".cache_weather"
MODEL_CHECKPOINT_DIR = "bk_model_checkpoints"

# --- 2. Mapování sloupců z Excelu ---
DATE_COLUMN = 'Date[Calendar Date]'
TIME_COLUMN = 'Check Closing Time[Check Closing Hour]'
SALES_COLUMN = '[CK___Sales_Net]'
GUESTS_COLUMN = '[CK___Guests]'
CHANNEL_COLUMN = 'Sales Channel[Sales Channel]'
LAT_COLUMN = 'Restaurant[Latitude]'
LON_COLUMN = 'Restaurant[Longitude]'

# --- 3. Konfigurace prognózy ---
TRAIN_START_DATE = '2023-01-01'
# Uprav dle potřeby
FORECAST_START = '2025-11-01 00:00:00'
FORECAST_END = '2025-11-30 23:00:00'
FREQ = 'h'

# --- 4. Parametry Modelů ---
# Nastavení pro tvůj HW (RTX 5070 + Ryzen)
TRAINER_KWARGS = {
    'accelerator': 'auto',  # Vynutí NVIDIA GPU
    'devices': 1,          # Použije 1 kartu
    'enable_model_summary': True,
    'strategy': 'auto'
}

TFT_PARAMS = {
    'h': 24,                # Horizont modelu (krok predikce)
    'input_size': 128,      # 7 dní zpět
    'max_steps': 1000,      # Hlavní parametr délky tréninku
    'early_stop_patience_steps': 15, # Zastaví, pokud se 15 kroků nezlepší
    'learning_rate': 0.001,
    'hidden_size': 64,
    'batch_size': 64,      # Zvýšeno pro RTX 5070
    'scaler_type': 'robust',
    'num_workers': 7
}

# --- 5. Seznam Features (Exogenní proměnné) ---
FUTR_EXOG_LIST = [
    'sin_hour', 'cos_hour',
    'sin_day', 'cos_day',
    'is_peak_lunch',
    'is_peak_dinner',
    'is_holiday',
    'is_weekend',
    'is_payday_week',
    'is_long_weekend',
    'is_school_holiday',
    'is_event_rfp',
    'is_event_vp',
    'is_event_ba',
    'is_event_ap',
    'is_competitor_closed'
]

# --- 6. Business Logika ---
PEAK_HOURS_LUNCH = [11, 12, 13, 14]
PEAK_HOURS_DINNER = [17, 18, 19, 20]

# --- 7. KALENDÁŘ UDÁLOSTÍ (EVENTS) ---
EVENT_KFC_CLOSED = ('2024-10-01', '2024-10-31')

EVENT_ROCK_FOR_PEOPLE = [
    ('2023-06-07', '2023-06-12'),
    ('2024-06-11', '2024-06-16'),
    ('2025-06-10', '2025-06-15')
]

EVENT_VELKA_PARDUBICKA = [
    ('2023-10-07', '2023-10-08'),
    ('2024-10-12', '2024-10-13'),
    ('2025-10-11', '2025-10-12')
]

EVENT_BRUTAL_ASSAULT = [
    ('2023-08-09', '2023-08-13'),
    ('2024-08-07', '2024-08-11'),
    ('2025-08-06', '2025-08-10')
]

EVENT_AVIATICKA_POUT = [
    ('2023-05-27', '2023-05-28'),
    ('2024-06-01', '2024-06-02'),
    ('2025-05-31', '2025-06-01')
]

SCHOOL_HOLIDAYS = [
    ('2023-02-06', '2023-03-19'), ('2023-04-06', '2023-04-10'),
    ('2023-07-01', '2023-09-03'), ('2023-10-26', '2023-10-29'),
    ('2023-12-23', '2024-01-02'),
    ('2024-02-05', '2024-03-17'), ('2024-03-28', '2024-04-01'),
    ('2024-06-29', '2024-09-01'), ('2024-10-29', '2024-10-30'),
    ('2024-12-23', '2025-01-05'),
    ('2025-02-03', '2025-03-16'), ('2025-04-17', '2025-04-21'),
    ('2025-06-28', '2025-08-31'), ('2025-10-29', '2025-10-30'),
    ('2025-12-22', '2026-01-04')
]

WEATHER_PARAMS = {
    "hourly": ["temperature_2m", "precipitation"],
    "timezone": "Europe/Berlin"
}