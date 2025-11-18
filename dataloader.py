import pandas as pd
import numpy as np
import config

class DataLoader:
    """
    Načítá data, čistí chyby (např. #N/V z Excelu), parsuje čas
    a vytváří kompletní časovou osu (včetně nulových hodin).
    """
    def __init__(self):
        pass

    def load_data(self):
        print(f"INFO: Načítám data ze souboru '{config.DATA_FILE}'...")
        try:
            df = pd.read_excel(config.DATA_FILE, sheet_name=config.SHEET_NAME)
        except Exception as e:
            print(f"CHYBA: Nelze načíst Excel: {e}")
            return pd.DataFrame(), pd.DataFrame(), None, None

        # --- 1. Čištění a Zpracování Času (ROBUSTNÍ FIX) ---
        print("INFO: Zpracovávám časové údaje a čistím chyby...")

        # Převedeme sloupec s časem na string
        time_col = df[config.TIME_COLUMN].astype(str).str.strip()

        # Vezmeme jen část před dvojtečkou (řeší "11:00" -> "11", ale i "11" -> "11")
        hour_part = time_col.str.split(':').str[0]

        # DŮLEŽITÉ: Bezpečný převod na číslo.
        # 'errors=coerce' změní '#N/V', 'null', 'text' na NaN (Not a Number)
        df['hour_int'] = pd.to_numeric(hour_part, errors='coerce')

        # Zjistíme, kolik řádků je vadných
        n_missing = df['hour_int'].isna().sum()
        if n_missing > 0:
            print(f"VAROVÁNÍ: Nalezeno {n_missing} řádků s neplatným časem (např. #N/V). Tyto řádky ignoruji.")
            # Odstraníme řádky, kde se nepovedlo určit hodinu
            df = df.dropna(subset=['hour_int'])

        # Teď už bezpečně převedeme na integer
        df['hour_int'] = df['hour_int'].astype(int)

        # Zpracování Data
        df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])

        # Vytvoření indexu 'ds' (Datum + Hodina)
        df['ds'] = df[config.DATE_COLUMN] + pd.to_timedelta(df['hour_int'], unit='h')

        # Přejmenování
        df = df.rename(columns={
            config.SALES_COLUMN: 'y_sales',
            config.GUESTS_COLUMN: 'y_guests',
            config.CHANNEL_COLUMN: 'unique_id',
            config.LAT_COLUMN: 'lat',
            config.LON_COLUMN: 'lon'
        })

        # Filtr historie
        df = df[df['ds'] >= config.TRAIN_START_DATE]

        # Získání souřadnic (pokud existují)
        lat_val = df['lat'].iloc[0] if 'lat' in df.columns and not df['lat'].isna().all() else None
        lon_val = df['lon'].iloc[0] if 'lon' in df.columns and not df['lon'].isna().all() else None

        # --- 2. Agregace duplicit ---
        # Sčítáme tržby, pokud je více řádků pro stejnou hodinu a kanál
        df_agg = df.groupby(['ds', 'unique_id']).agg({
            'y_sales': 'sum', 'y_guests': 'sum'
        }).reset_index()

        # --- 3. Vytvoření Totalu ---
        df_total = df_agg.groupby('ds').agg({
            'y_sales': 'sum', 'y_guests': 'sum'
        }).reset_index()
        df_total['unique_id'] = 'Total'

        combined_df = pd.concat([df_agg, df_total], ignore_index=True)

        # --- 4. FILL ZEROS (Doplnění chybějících hodin) ---
        # Kritické pro časové řady: Pokud v datech chybí hodina (zavřeno), musí tam být nula.
        print("INFO: Doplňuji nulové prodeje pro hodiny, kdy bylo zavřeno...")

        unique_ids = combined_df['unique_id'].unique()

        # Rozsah od začátku do konce dat
        if combined_df.empty:
            print("CHYBA: Po vyčištění nezbyla žádná data!")
            return pd.DataFrame(), pd.DataFrame(), None, None

        full_idx = pd.date_range(start=combined_df['ds'].min(), end=combined_df['ds'].max(), freq='h')

        final_dfs = []
        for uid in unique_ids:
            # Vezmeme data jednoho kanálu
            temp = combined_df[combined_df['unique_id'] == uid].set_index('ds')

            # Reindexujeme na plný rozsah hodin -> vzniknou NaN tam, kde bylo zavřeno
            temp_reindexed = temp.reindex(full_idx).fillna({
                'y_sales': 0,
                'y_guests': 0,
                'unique_id': uid # Vrátíme ID zpět
            }).reset_index().rename(columns={'index': 'ds'})

            final_dfs.append(temp_reindexed)

        final_df = pd.concat(final_dfs, ignore_index=True)

        # Rozdělení výstupů
        df_sales = final_df[['ds', 'unique_id', 'y_sales']].rename(columns={'y_sales': 'y'})
        df_guests = final_df[['ds', 'unique_id', 'y_guests']].rename(columns={'y_guests': 'y'})

        print(f"INFO: Data připravena. Počet řádků: {len(final_df)}")
        return df_sales, df_guests, lat_val, lon_val

    @staticmethod
    def get_hierarchy_matrix(df):
        # Matice pro rekonciliaci (BottomUp)
        unique_ids = df[df['unique_id'] != 'Total']['unique_id'].unique()
        unique_ids.sort()
        S = pd.DataFrame(0, index=['Total'] + list(unique_ids), columns=list(unique_ids))
        S.loc['Total'] = 1
        for uid in unique_ids:
            S.loc[uid, uid] = 1

        tags = {
            'Total': ['Total'],
            'Total/Channel': list(unique_ids)
        }
        return S, tags