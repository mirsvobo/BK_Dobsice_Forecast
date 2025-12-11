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
            print(f"CHYBA: Nelze načíst soubor: {e}")
            return pd.DataFrame(), pd.DataFrame(), None, None

        # 1. Očištění a přejmenování
        df.columns = [c.strip() for c in df.columns]

        rename_map = {
            config.DATE_COLUMN: 'date_raw',
            config.SALES_COLUMN: 'y_sales',
            config.GUESTS_COLUMN: 'y_guests',
            config.CHANNEL_COLUMN: 'unique_id'
        }

        # Aplikace mapy jen na existující sloupce
        final_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=final_map)

        # GPS z Configu
        lat = config.RESTAURANT_META['Latitude']
        lon = config.RESTAURANT_META['Longitude']

        # 2. Zpracování data
        df['ds'] = pd.to_datetime(df['date_raw'], errors='coerce')
        df = df.dropna(subset=['ds'])

        # Konverze čísel
        for col in ['y_sales', 'y_guests']:
            if col in df.columns:
                if df[col].dtype == object:
                    # Odstraníme čárky i mezery (např. "1 200")
                    df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0

        # --- [FIX] SANITIZACE DAT (Záporná čísla rozbíjejí TFT) ---
        # Záporné prodeje (vratky) ořízneme na 0, jinak model diverguje (MAE: inf)
        num_neg_sales = (df['y_sales'] < 0).sum()
        if num_neg_sales > 0:
            print(f"⚠️ VAROVÁNÍ: Nalezeno {num_neg_sales} řádků se zápornými tržbami. Ořezávám na 0.")
            df['y_sales'] = df['y_sales'].clip(lower=0.0)

        # To samé pro hosty (logicky nemůže být záporný počet hostů)
        df['y_guests'] = df['y_guests'].clip(lower=0.0)
        # -----------------------------------------------------------

        # 3. AGREGACE NA DNY
        print("INFO: Agreguji data na denní úroveň...")
        df_agg = df.groupby(['ds', 'unique_id'])[['y_sales', 'y_guests']].sum().reset_index()
        df = df_agg

        # --- FIX: DOPLNĚNÍ CHYBĚJÍCÍCH DNŮ (GAPS) ---
        if not df.empty:
            all_dates = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
            all_ids = df['unique_id'].unique()

            idx = pd.MultiIndex.from_product([all_dates, all_ids], names=['ds', 'unique_id'])
            df = df.set_index(['ds', 'unique_id']).reindex(idx, fill_value=0).reset_index()

        # 4. Filtr historických dat
        mask = (df['ds'] >= config.TRAIN_START_DATE)
        df = df.loc[mask]

        # 5. Formát pro model
        sales_df = df[['ds', 'unique_id', 'y_sales']].rename(columns={'y_sales': 'y'})
        guests_df = df[['ds', 'unique_id', 'y_guests']].rename(columns={'y_guests': 'y'})

        # 6. Total kanál
        sales_total = sales_df.groupby('ds')['y'].sum().reset_index()
        sales_total['unique_id'] = 'Total'
        sales_df = pd.concat([sales_df, sales_total])

        guests_total = guests_df.groupby('ds')['y'].sum().reset_index()
        guests_total['unique_id'] = 'Total'
        guests_df = pd.concat([guests_df, guests_total])

        # Seřazení
        sales_df = sales_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        guests_df = guests_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

        print(f"INFO: Data připravena. Počet řádků: {len(sales_df)}")
        return sales_df, guests_df, lat, lon

    @staticmethod
    def get_hierarchy_matrix(df):
        unique_ids = df['unique_id'].unique()
        S = pd.DataFrame(np.eye(len(unique_ids)), index=unique_ids, columns=unique_ids)
        tags = {'Country': unique_ids, 'Country/State': unique_ids}
        return S, tags