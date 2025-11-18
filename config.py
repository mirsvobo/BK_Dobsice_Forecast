import os

# --- 1. Konfigurace cest a souborů ---
DATA_FILE = 'data.xlsx'
SHEET_NAME = "OPSBKSalesMixOverview4"
WEATHER_CACHE_DIR = ".cache_weather"

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
FORECAST_START = '2025-11-01 00:00:00'
FORECAST_END = '2025-11-30 23:00:00'
FREQ = 'h'  # Hodinová frekvence

# --- 4. Parametry Modelů ---
CHRONOS_MODEL = "amazon/chronos-t5-tiny"

TFT_PARAMS = {
    'h': 24,
    'input_size': 168,      # 7 dní zpět (7 * 24h)
    'max_steps': 2000,      # Počet kroků učení
    'learning_rate': 0.001,
    'hidden_size': 64,
    'batch_size': 64,
    'scaler_type': 'robust'
}

# --- 5. Seznam Features (Exogenní proměnné) ---
# Tyto sloupce musí FeatureEngineer vyrobit a model je očekává
FUTR_EXOG_LIST = [
    'sin_hour', 'cos_hour',       # Denní cyklus
    'sin_day', 'cos_day',         # Týdenní cyklus
    'is_peak_lunch',              # Oběd
    'is_peak_dinner',             # Večeře
    'is_holiday',                 # Svátek
    'is_weekend',                 # Víkend
    'is_payday_week',             # Výplata
    'is_long_weekend',            # Prodloužený víkend
    'is_school_holiday',          # Prázdniny
    'is_event_rfp',               # Rock for People
    'is_event_vp',                # Velká pardubická
    'is_event_ba',                # Brutal Assault
    'is_event_ap',                # Aviatická pouť
    'is_competitor_closed'        # KFC rekonstrukce
]

# --- 6. Business Logika ---
PEAK_HOURS_LUNCH = [11, 12, 13, 14]
PEAK_HOURS_DINNER = [17, 18, 19, 20]

# --- 7. KALENDÁŘ UDÁLOSTÍ (EVENTS) ---

# A. KFC Closed (Rekonstrukce konkurence)
EVENT_KFC_CLOSED = ('2024-10-01', '2024-10-31')

# B. Rock for People (Hradec Králové)
EVENT_ROCK_FOR_PEOPLE = [
    ('2023-06-07', '2023-06-12'),
    ('2024-06-11', '2024-06-16'),
    ('2025-06-10', '2025-06-15')
]

# C. Velká pardubická (Pardubice - Massive Traffic z Prahy)
EVENT_VELKA_PARDUBICKA = [
    ('2023-10-07', '2023-10-08'),
    ('2024-10-12', '2024-10-13'),
    ('2025-10-11', '2025-10-12')
]

# D. Brutal Assault (Jaroměř - Pevnost Josefov, D11 Tranzit)
EVENT_BRUTAL_ASSAULT = [
    ('2023-08-09', '2023-08-13'),
    ('2024-08-07', '2024-08-11'),
    ('2025-08-06', '2025-08-10')
]

# E. Aviatická pouť (Pardubice)
EVENT_AVIATICKA_POUT = [
    ('2023-05-27', '2023-05-28'),
    ('2024-06-01', '2024-06-02'),
    ('2025-05-31', '2025-06-01')
]

# F. Školní prázdniny (Pohyby rodin)
SCHOOL_HOLIDAYS = [
    # 2023
    ('2023-02-06', '2023-03-19'), ('2023-04-06', '2023-04-10'),
    ('2023-07-01', '2023-09-03'), ('2023-10-26', '2023-10-29'),
    ('2023-12-23', '2024-01-02'),
    # 2024
    ('2024-02-05', '2024-03-17'), ('2024-03-28', '2024-04-01'),
    ('2024-06-29', '2024-09-01'), ('2024-10-29', '2024-10-30'),
    ('2024-12-23', '2025-01-05'),
    # 2025
    ('2025-02-03', '2025-03-16'), ('2025-04-17', '2025-04-21'),
    ('2025-06-28', '2025-08-31'), ('2025-10-29', '2025-10-30'),
    ('2025-12-22', '2026-01-04')
]

# Nastavení pro počasí
WEATHER_PARAMS = {
    "hourly": ["temperature_2m", "precipitation"],
    "timezone": "Europe/Berlin"
}