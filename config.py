import os

# --- 1. Konfigurace cest a souborů ---
DATA_FILE = 'data1.xlsx'  # Nový soubor
SHEET_NAME = 0            # Načteme první list (obvykle index 0)
WEATHER_CACHE_DIR = ".cache_weather"
MODEL_CHECKPOINT_DIR = "bk_model_checkpoints"

# --- 2. ATRIBUTY RESTAURACE (Hardcoded) ---
# Převedeno ze zadání přímo do kódu
RESTAURANT_META = {
    'City': 'Dobsice',
    'Latitude': 50.121113,
    'Longitude': 15.279772,
    'Name': 'BK Dobsice D11',
    'Opening_Date': '2021-02-04',
    'Kiosk_Date': '2022-01-29',
    'Delivery_Date': '2021-03-12'
}

# --- 3. Mapování sloupců (dle data1.xlsx) ---
# Na základě vašeho CSV exportu:
# "Calendar Date", "CK - Sales Net", "Sales Channel", "CK - Guests"
DATE_COLUMN = 'Calendar Date'
SALES_COLUMN = 'CK - Sales Net'
GUESTS_COLUMN = 'CK - Guests'
CHANNEL_COLUMN = 'Sales Channel'

# --- 4. Konfigurace prognózy (DENNÍ) ---
TRAIN_START_DATE = '2021-02-01' # Od otevření
FORECAST_START = '2025-11-01'
FORECAST_END = '2025-11-30'
FREQ = 'D'  # <--- DENNÍ FREKVENCE

# --- 5. Parametry modelu (TFT - Daily) ---
TFT_PARAMS = {
    'h': 31,                # 31 dní dopředu
    'input_size': 60,       # 2 měsíce kontextu
    'max_steps': 500,
    'early_stop_patience_steps': 20,
    'learning_rate': 0.001,
    'hidden_size': 64,
    'batch_size': 64,       # Bezpečné pro paměť
    'scaler_type': 'robust',
}

TRAINER_KWARGS = {
    'accelerator': 'gpu',
    'devices': 1,
}

# --- 6. Features (Denní) ---
FUTR_EXOG_LIST = [
    'sin_day', 'cos_day',
    'is_holiday', 'is_weekend',
    'is_payday_week', 'is_long_weekend',
    'is_school_holiday',
    'is_event_rfp', 'is_event_vp', 'is_event_ba',
    'is_event_ap', 'is_competitor_closed',
    'is_covid_restriction'
]

# --- 7. Business Logika ---
# Eventy
EVENT_KFC_CLOSED = ('2024-10-01', '2024-10-31')
EVENT_ROCK_FOR_PEOPLE = [('2023-06-07', '2023-06-12'), ('2024-06-11', '2024-06-16'), ('2025-06-10', '2025-06-15')]
EVENT_VELKA_PARDUBICKA = [('2023-10-07', '2023-10-08'), ('2024-10-12', '2024-10-13'), ('2025-10-11', '2025-10-12')]
EVENT_BRUTAL_ASSAULT = [('2023-08-09', '2023-08-13'), ('2024-08-07', '2024-08-11'), ('2025-08-06', '2025-08-10')]
EVENT_AVIATICKA_POUT = [('2023-05-27', '2023-05-28'), ('2024-06-01', '2024-06-02'), ('2025-05-31', '2025-06-01')]
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
COVID_RESTRICTIONS = [
    ('2020-12-18', '2021-05-16'),
    ('2021-05-17', '2021-06-30'),
    ('2021-11-26', '2021-12-25'),
    ('2022-01-01', '2022-02-18')
]

WEATHER_PARAMS = {
    "hourly": ["temperature_2m", "precipitation"],
    "timezone": "Europe/Berlin"
}