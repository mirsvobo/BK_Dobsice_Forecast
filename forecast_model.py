import pandas as pd
import numpy as np
import os
import shutil
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE

class ForecastModel:
    def __init__(self):
        self.nf_sales = None
        self.nf_guests = None

        # --- KONFIGURACE ---
        self.config = {
            "max_steps": 2500,           # Dostatek kroků pro učení
            "learning_rate": 0.0005,     # Přesnější učení
            "batch_size": 4096,          # ⚠️ Bezpečná hodnota pro VRAM i při delším horizontu
            "val_check_steps": 100,
            "early_stop_patience_steps": 30,
            "scaler_type": 'standard',
            "enable_progress_bar": False
        }

    def train(self, sales_df, guests_df, horizon, callbacks_sales=None, callbacks_guests=None):
        """
        Trénuje model na specifický horizont (počet dní).
        """
        print(f"🚀 Začínám trénink na {horizon} dní (Batch={self.config['batch_size']})...")

        # Input size (kolik historie vidí) = 3x horizont
        input_size = 3 * horizon

        # --- 1. MODEL TRŽBY ---
        if callbacks_sales:
            sales_std = sales_df['y'].std()
            for cb in callbacks_sales:
                cb.y_std = sales_std

        models_sales = [
            NHITS(
                h=horizon,               # Nastavíme horizont dynamicky
                input_size=input_size,
                loss=MAE(),
                callbacks=callbacks_sales if callbacks_sales else [],
                **self.config
            )
        ]

        self.nf_sales = NeuralForecast(models=models_sales, freq='D')
        self.nf_sales.fit(df=sales_df, val_size=horizon)

        # --- 2. MODEL HOSTÉ ---
        if callbacks_guests:
            guests_std = guests_df['y'].std()
            for cb in callbacks_guests:
                cb.y_std = guests_std

        models_guests = [
            NHITS(
                h=horizon,               # Nastavíme horizont dynamicky
                input_size=input_size,
                loss=MAE(),
                callbacks=callbacks_guests if callbacks_guests else [],
                **self.config
            )
        ]

        self.nf_guests = NeuralForecast(models=models_guests, freq='D')
        self.nf_guests.fit(df=guests_df, val_size=horizon)

        print("✅ Trénink dokončen.")

    def predict(self, future_df_with_weather, S=None, tags=None):
        if self.nf_sales is None or self.nf_guests is None:
            raise ValueError("Modely nejsou natrénované!")

        # Model si sám vytvoří dataframe o délce svého horizontu 'h'
        futr_sales = self.nf_sales.make_future_dataframe()

        # Připojíme počasí
        # (Ošetření: weather data musí pokrývat celou dobu)
        weather_data = future_df_with_weather.drop(columns=['unique_id'], errors='ignore').drop_duplicates('ds')
        futr_sales = futr_sales.merge(weather_data, on='ds', how='left')

        preds_sales = self.nf_sales.predict(futr_df=futr_sales)

        # Totéž pro hosty
        futr_guests = self.nf_guests.make_future_dataframe()
        futr_guests = futr_guests.merge(weather_data, on='ds', how='left')
        preds_guests = self.nf_guests.predict(futr_df=futr_guests)

        def get_model_col(df):
            candidates = [c for c in df.columns if c not in ['ds', 'unique_id', 'y']]
            return candidates[0] if candidates else None

        col_sales = get_model_col(preds_sales)
        col_guests = get_model_col(preds_guests)

        preds_sales['Forecast_Value'] = preds_sales[col_sales]
        preds_guests['Forecast_Value'] = preds_guests[col_guests]

        return preds_sales, preds_guests

    def save_model(self, path):
        if os.path.exists(path):
            # Pro jistotu smažeme staré, aby se nepomíchaly verze
            shutil.rmtree(path)
        os.makedirs(path)

        self.nf_sales.save(os.path.join(path, "sales_model"), overwrite=True)
        self.nf_guests.save(os.path.join(path, "guests_model"), overwrite=True)
        print(f"💾 Modely uloženy do {path}")

    def load_model(self, path, required_horizon):
        """
        Načte model jen pokud existuje A pokud má dostatečný horizont.
        """
        sales_path = os.path.join(path, "sales_model")
        guests_path = os.path.join(path, "guests_model")

        if os.path.exists(sales_path) and os.path.exists(guests_path):
            try:
                print("📂 Kontroluji uložené modely...")
                temp_sales = NeuralForecast.load(sales_path)

                # ZJISTÍME HORIZONT ULOŽENÉHO MODELU
                # NeuralForecast drží seznam modelů, vezmeme první
                stored_h = temp_sales.models[0].h

                if stored_h < required_horizon:
                    print(f"⚠️ Uložený model má krátký horizont ({stored_h} dní). Požadováno {required_horizon}. Je nutný přetrénink.")
                    return False

                self.nf_sales = temp_sales
                self.nf_guests = NeuralForecast.load(guests_path)
                print(f"✅ Modely načteny (Horizont: {stored_h} dní).")
                return True
            except Exception as e:
                print(f"⚠️ Chyba při načítání modelu: {e}")
                return False
        else:
            return False