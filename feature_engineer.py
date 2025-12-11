import pandas as pd
import numpy as np
import holidays
import config

class FeatureEngineer:
    def __init__(self):
        self.cz_holidays = holidays.CZ()

    def transform(self, df, weather_df=None):
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])

        # 1. Merge Weather (Optimized join)
        if weather_df is not None:
            weather_df['ds'] = pd.to_datetime(weather_df['ds'])
            # Ensure weather is daily mean
            weather_daily = weather_df.resample('D', on='ds').mean().reset_index()
            # Left join is faster if keys are sorted
            df = pd.merge(df, weather_daily, on='ds', how='left')

            cols = [c for c in weather_df.columns if c != 'ds']
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].ffill().fillna(0).astype(np.float32)

        # 2. Time Features (Vectorized)
        df['dayofweek'] = df['ds'].dt.dayofweek.astype(np.int8)
        df['day'] = df['ds'].dt.day.astype(np.int8)

        # 3. Cyclic Encoding
        df['sin_day'] = np.sin(2 * np.pi * df['dayofweek'] / 7).astype(np.float32)
        df['cos_day'] = np.cos(2 * np.pi * df['dayofweek'] / 7).astype(np.float32)

        # 4. Boolean Flags (Int8 is sufficient)
        df['is_holiday'] = df['ds'].dt.date.isin(self.cz_holidays).astype(np.int8)
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(np.int8)
        df['is_payday_week'] = ((df['day'] >= 10) & (df['day'] <= 15)).astype(np.int8)

        # Long Weekend Logic
        df['is_long_weekend'] = 0
        mask_fri_hol = (df['dayofweek'] == 4) & (df['is_holiday'] == 1)
        mask_mon_hol = (df['dayofweek'] == 0) & (df['is_holiday'] == 1)
        df.loc[mask_fri_hol | mask_mon_hol, 'is_long_weekend'] = 1
        df['is_long_weekend'] = df['is_long_weekend'].astype(np.int8)

        # 5. Events - Helper
        def apply_event(df, periods, col_name):
            df[col_name] = 0
            if not periods: return
            # Pre-calculate mask for speed
            mask = pd.Series(False, index=df.index)
            for start, end in periods:
                s = pd.Timestamp(start)
                e = pd.Timestamp(end)
                mask |= (df['ds'] >= s) & (df['ds'] <= e)
            df.loc[mask, col_name] = 1
            df[col_name] = df[col_name].astype(np.int8)

        apply_event(df, config.EVENT_ROCK_FOR_PEOPLE, 'is_event_rfp')
        apply_event(df, config.EVENT_VELKA_PARDUBICKA, 'is_event_vp')
        apply_event(df, config.EVENT_BRUTAL_ASSAULT, 'is_event_ba')
        apply_event(df, config.EVENT_AVIATICKA_POUT, 'is_event_ap')
        apply_event(df, config.SCHOOL_HOLIDAYS, 'is_school_holiday')
        apply_event(df, config.COVID_RESTRICTIONS, 'is_covid_restriction')
        apply_event(df, config.CHRISTMAS_CLOSED, 'is_closed')
        apply_event(df, config.CHRISTMAS_SHORT, 'is_short_open')

        # Competitor
        df['is_competitor_closed'] = 0
        if hasattr(config, 'EVENT_KFC_CLOSED'):
            s = pd.Timestamp(config.EVENT_KFC_CLOSED[0])
            e = pd.Timestamp(config.EVENT_KFC_CLOSED[1])
            mask = (df['ds'] >= s) & (df['ds'] <= e)
            df.loc[mask, 'is_competitor_closed'] = 1
        df['is_competitor_closed'] = df['is_competitor_closed'].astype(np.int8)

        return df