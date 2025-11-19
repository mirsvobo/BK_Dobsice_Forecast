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
            # Načtení Excelu (nebo CSV, pandas to zvládne detekovat, pokud koncovka sedí)
            if self.file_path.endswith('.csv'):
                df = pd.read_csv(self.file_path)
            else:
                df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        except Exception as e:
            print(f"CHYBA: Nelze načíst soubor: {e}")
            return pd.DataFrame(), pd.DataFrame(), None, None

        # 1. Přejmenování sloupců dle Configu
        # Používáme nové mapování pro data1.xlsx
        rename_map = {
            config.DATE_COLUMN: 'date_raw',
            config.SALES_COLUMN: 'y_sales',
            config.GUESTS_COLUMN: 'y_guests',
            config.CHANNEL_COLUMN: 'unique_id'
        }
        df = df.rename(columns=rename_map)

        # GPS souřadnice - BEREME Z CONFIGU (Hardcoded atributy)
        lat = config.RESTAURANT_META['Latitude']
        lon = config.RESTAURANT_META['Longitude']

        # 2. Zpracování data
        df['ds'] = pd.to_datetime(df['date_raw'])
        df = df.dropna(subset=['ds'])

        # Konverze čísel (ošetření čárek/teček)
        for col in ['y_sales', 'y_guests']:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 3. AGREGACE NA DNY (Sumarizace)
        # Protože v souboru mohou být řádky pro stejný den a kanál (nebo hodiny),
        # sečteme je na úroveň Dne.
        print("INFO: Agreguji data na denní úroveň...")
        df_agg = df.groupby(['ds', 'unique_id'])[['y_sales', 'y_guests']].sum().reset_index()
        df = df_agg

        # 4. Filtr historických dat
        mask = (df['ds'] >= config.TRAIN_START_DATE)
        df = df.loc[mask]

        # 5. Příprava formátu
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

        print(f"INFO: Data připravena. Počet řádků (Daily): {len(sales_df)}")
        return sales_df, guests_df, lat, lon

    @staticmethod
    def get_hierarchy_matrix(df):
        unique_ids = df['unique_id'].unique()
        S = pd.DataFrame(np.eye(len(unique_ids)), index=unique_ids, columns=unique_ids)
        tags = {'Country': unique_ids, 'Country/State': unique_ids}
        return S, tags