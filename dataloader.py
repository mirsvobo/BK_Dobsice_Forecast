import pandas as pd
import numpy as np
import config

class DataLoader:
    def __init__(self):
        self.file_path = config.DATA_FILE
        self.sheet_name = config.SHEET_NAME

    def load_data(self):
        print(f"INFO: Načítám data ze souboru '{self.file_path}'...")
        try:
            if self.file_path.lower().endswith('.csv'):
                df = pd.read_csv(self.file_path)
            else:
                df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        except Exception as e:
            print(f"CRITICAL ERROR: Nelze načíst soubor: {e}")
            return pd.DataFrame(), pd.DataFrame(), None, None

        # Cleaning Column Names
        df.columns = [c.strip() for c in df.columns]

        # Rename Map
        rename_map = {
            config.DATE_COLUMN: 'date_raw',
            config.SALES_COLUMN: 'y_sales',
            config.GUESTS_COLUMN: 'y_guests',
            config.CHANNEL_COLUMN: 'unique_id'
        }
        final_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=final_map)

        # GPS
        lat = config.RESTAURANT_META['Latitude']
        lon = config.RESTAURANT_META['Longitude']

        # Date Parsing
        df['ds'] = pd.to_datetime(df['date_raw'], errors='coerce')
        df = df.dropna(subset=['ds'])

        # Numeric Conversion & Cleaning
        for col in ['y_sales', 'y_guests']:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # Clip negative values
                df[col] = df[col].clip(lower=0.0)
            else:
                df[col] = 0.0

        # Aggregation to Daily
        print("INFO: Agreguji data na denní úroveň...")
        df_agg = df.groupby(['ds', 'unique_id'])[['y_sales', 'y_guests']].sum().reset_index()
        df = df_agg

        # Fill Gaps (Important for Time Series Models)
        if not df.empty:
            all_dates = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
            all_ids = df['unique_id'].unique()
            idx = pd.MultiIndex.from_product([all_dates, all_ids], names=['ds', 'unique_id'])
            df = df.set_index(['ds', 'unique_id']).reindex(idx, fill_value=0).reset_index()

        # Filter History
        mask = (df['ds'] >= config.TRAIN_START_DATE)
        df = df.loc[mask]

        # Create Total Channel
        sales_total = df.groupby('ds')['y_sales'].sum().reset_index()
        sales_total['unique_id'] = 'Total'
        guests_total = df.groupby('ds')['y_guests'].sum().reset_index()
        guests_total['unique_id'] = 'Total'

        # Final Dataframes
        sales_df = pd.concat([df[['ds', 'unique_id', 'y_sales']].rename(columns={'y_sales': 'y'}),
                              sales_total.rename(columns={'y_sales': 'y'})])

        guests_df = pd.concat([df[['ds', 'unique_id', 'y_guests']].rename(columns={'y_guests': 'y'}),
                               guests_total.rename(columns={'y_guests': 'y'})])

        # Optimize Types for GPU (float32 is faster than float64)
        sales_df['y'] = sales_df['y'].astype(np.float32)
        guests_df['y'] = guests_df['y'].astype(np.float32)

        sales_df = sales_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        guests_df = guests_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

        print(f"INFO: Data připravena. Rows: {len(sales_df)}")
        return sales_df, guests_df, lat, lon

    @staticmethod
    def get_hierarchy_matrix(df):
        unique_ids = df['unique_id'].unique()
        S = pd.DataFrame(np.eye(len(unique_ids)), index=unique_ids, columns=unique_ids)
        tags = {'Country': unique_ids, 'Country/State': unique_ids}
        return S, tags