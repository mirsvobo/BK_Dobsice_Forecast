import pandas as pd
from datetime import timedelta

# --- 1. GLOBÁLNÍ NASTAVENÍ (Výchozí) ---
FORECAST_HORIZON_DAYS = 31

# Cesty
DATA_FILE = 'data1.xlsx'
SHEET_NAME = 0
WEATHER_CACHE_DIR = ".cache_weather"
MODEL_CHECKPOINT_DIR = "bk_model_checkpoints"

# --- 2. DYNAMICKÉ DATA ---
# Toto se bude aktualizovat v dashboardu, ale pro inicializaci:
TODAY = pd.Timestamp.now().normalize()
FORECAST_START = TODAY + timedelta(days=1)
# Default end
FORECAST_END = TODAY + timedelta(days=FORECAST_HORIZON_DAYS)

TRAIN_START_DATE = '2021-02-01'
FREQ = 'D'

# --- 3. ATRIBUTY RESTAURACE ---
RESTAURANT_META = {
    'City': 'Dobsice',
    'Latitude': 50.121113,
    'Longitude': 15.279772,
    'Name': 'BK Dobsice D11'
}

# --- 4. MAPOVÁNÍ SLOUPCŮ ---
DATE_COLUMN = 'Calendar Date'
SALES_COLUMN = 'CK - Sales Net'
GUESTS_COLUMN = 'CK - Guests'
CHANNEL_COLUMN = 'Sales Channel'

# --- 5. PARAMETRY MODELU ---
# Optimalizováno pro RTX karty
TFT_PARAMS = {
    'h': FORECAST_HORIZON_DAYS,
    'input_size': 60,                # Kratší historie pro trénink = rychlejší epochy
    'max_steps': 400,                # Dostatečné pro konvergenci na tomto datasetu
    'learning_rate': 0.001,
    'hidden_size': 64,               # 64-128 je sweet spot pro rychlost/výkon
    'batch_size': 64,
    'scaler_type': 'robust',
    'dropout': 0.1,
    'attn_head_size': 4,
}

# --- 6. EXTERNÍ VLIVY ---
FUTR_EXOG_LIST = [
    'sin_day', 'cos_day',
    'is_holiday', 'is_weekend',
    'is_payday_week', 'is_long_weekend',
    'is_school_holiday',
    'is_event_rfp', 'is_event_vp', 'is_event_ba',
    'is_event_ap', 'is_competitor_closed',
    'is_covid_restriction',
    'is_closed', 'is_short_open',
    'temperature_2m', 'precipitation'
]

# --- 7. BUSINESS LOGIKA ---
# Generátor Vánoc
CHRISTMAS_CLOSED = []
CHRISTMAS_SHORT = []
for year in range(2021, 2027):
    CHRISTMAS_CLOSED.append((f'{year}-12-24', f'{year}-12-25'))
    CHRISTMAS_SHORT.append((f'{year}-12-26', f'{year}-12-26'))
    CHRISTMAS_SHORT.append((f'{year}-12-31', f'{year}-12-31'))

EVENT_KFC_CLOSED = ('2024-10-01', '2024-10-31')
EVENT_ROCK_FOR_PEOPLE = [('2023-06-07', '2023-06-12'), ('2024-06-11', '2024-06-16'), ('2025-06-10', '2025-06-15')]
EVENT_VELKA_PARDUBICKA = [('2023-10-07', '2023-10-08'), ('2024-10-12', '2024-10-13'), ('2025-10-11', '2025-10-12')]
EVENT_BRUTAL_ASSAULT = [('2023-08-09', '2023-08-13'), ('2024-08-07', '2024-08-11'), ('2025-08-06', '2025-08-10')]
EVENT_AVIATICKA_POUT = [('2023-05-27', '2023-05-28'), ('2024-06-01', '2024-06-02'), ('2025-05-31', '2025-06-01')]

SCHOOL_HOLIDAYS = [
    ('2023-07-01', '2023-09-03'), ('2023-12-23', '2024-01-02'),
    ('2024-06-29', '2024-09-01'), ('2024-12-23', '2025-01-05'),
    ('2025-02-03', '2025-03-16'), ('2025-06-28', '2025-08-31'),
    ('2025-12-22', '2026-01-04')
]
COVID_RESTRICTIONS = [('2020-12-18', '2021-05-16'), ('2021-11-26', '2021-12-25')]

WEATHER_PARAMS = {
    "hourly": ["temperature_2m", "precipitation"],
    "timezone": "Europe/Berlin"
}