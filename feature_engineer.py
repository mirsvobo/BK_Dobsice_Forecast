import pandas as pd
import numpy as np
import holidays
import config

class FeatureEngineer:
    """
    Vytváří features pro model:
    - Časové cykly (hodina, den)
    - Business špičky (oběd, večeře)
    - Kalendářní události (svátky, víkendy, výplaty)
    - Speciální eventy (Rock for People, Velká pardubická...)
    - NOVĚ: Lagované proměnné (Autoregrese)
    """
    def __init__(self):
        self.cz_holidays = holidays.CZ()

    def transform(self, df, weather_df=None):
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])

        # 1. Merge Weather (Počasí)
        if weather_df is not None:
            weather_df['ds'] = pd.to_datetime(weather_df['ds'])
            df = pd.merge(df, weather_df, on='ds', how='left')
            # Doplnění chybějících dat počasí (Forward Fill)
            cols = [c for c in weather_df.columns if c != 'ds']
            for c in cols:
                if c in df.columns: df[c] = df[c].fillna(method='ffill').fillna(0)

        # 2. Basic Time Features
        df['dayofweek'] = df['ds'].dt.dayofweek
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['hour'] = df['ds'].dt.hour

        # 3. Cyclic Encoding
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_day'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['cos_day'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # 4. Business Peaks
        df['is_peak_lunch'] = df['hour'].apply(lambda x: 1 if x in config.PEAK_HOURS_LUNCH else 0)
        df['is_peak_dinner'] = df['hour'].apply(lambda x: 1 if x in config.PEAK_HOURS_DINNER else 0)

        # 5. Svátky a Víkendy
        df['is_holiday'] = df['ds'].dt.date.apply(lambda x: 1 if x in self.cz_holidays else 0)
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_payday_week'] = df['day'].apply(lambda x: 1 if 10 <= x <= 15 else 0)

        # Prodloužený víkend
        df['is_long_weekend'] = 0
        mask_fri_hol = (df['dayofweek'] == 4) & (df['is_holiday'] == 1)
        mask_mon_hol = (df['dayofweek'] == 0) & (df['is_holiday'] == 1)
        df.loc[mask_fri_hol | mask_mon_hol, 'is_long_weekend'] = 1

        # --- 6. SPECIÁLNÍ UDÁLOSTI (EVENTS) ---
        def mark_event(dataframe, event_list, col_name):
            dataframe[col_name] = 0
            for start, end in event_list:
                s_date = pd.to_datetime(start).date()
                e_date = pd.to_datetime(end).date()
                mask = (dataframe['ds'].dt.date >= s_date) & (dataframe['ds'].dt.date <= e_date)
                dataframe.loc[mask, col_name] = 1
            return dataframe

        # Aplikace eventů z Configu
        df = mark_event(df, config.EVENT_ROCK_FOR_PEOPLE, 'is_event_rfp')
        df = mark_event(df, config.EVENT_VELKA_PARDUBICKA, 'is_event_vp')
        df = mark_event(df, config.EVENT_BRUTAL_ASSAULT, 'is_event_ba')
        df = mark_event(df, config.EVENT_AVIATICKA_POUT, 'is_event_ap')
        df = mark_event(df, config.SCHOOL_HOLIDAYS, 'is_school_holiday')

        # KFC Closed
        df['is_competitor_closed'] = 0
        if hasattr(config, 'EVENT_KFC_CLOSED'):
            kfc_start = pd.to_datetime(config.EVENT_KFC_CLOSED[0]).date()
            kfc_end = pd.to_datetime(config.EVENT_KFC_CLOSED[1]).date()
            mask_kfc = (df['ds'].dt.date >= kfc_start) & (df['ds'].dt.date <= kfc_end)
            df.loc[mask_kfc, 'is_competitor_closed'] = 1

        return df

    def create_lags(self, df):
        """
        Přidá zpožděné proměnné (Lags).
        Užitečné pro analýzu a robustnost, i když TFT má interní paměť.
        Používat opatrně na Future Data (tam lags nebudou existovat, vznikne NaN).
        """
        df = df.copy()
        df = df.sort_values(['unique_id', 'ds'])

        # 1. Lag 24h (Včera ve stejnou hodinu)
        df['lag_24h'] = df.groupby('unique_id')['y'].shift(24)

        # 2. Lag 168h (Minulý týden ve stejnou hodinu - VELMI SILNÉ)
        df['lag_168h'] = df.groupby('unique_id')['y'].shift(168)

        # Doplnění NaN na začátku (Backfill)
        df['lag_24h'] = df['lag_24h'].fillna(method='bfill').fillna(0)
        df['lag_168h'] = df['lag_168h'].fillna(method='bfill').fillna(0)

        return df