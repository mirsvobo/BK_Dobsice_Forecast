import pandas as pd
import numpy as np
import os
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import HuberMQLoss
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp
import config
import warnings
import logging
import torch # NOVÝ IMPORT PRO KONTROLU CUDA

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

class ForecastModel:
    def __init__(self, best_params=None):
        self.nf_sales = None
        self.nf_guests = None
        self.reconciler = HierarchicalReconciliation(reconcilers=[BottomUp()])
        self.loss = HuberMQLoss(quantiles=[0.1, 0.5, 0.9], delta=1.0)
        self.params = best_params if best_params else config.TFT_PARAMS

    def _build_models_list(self):
        h_size = self.params.get('hidden_size', config.TFT_PARAMS['hidden_size'])
        lr = self.params.get('learning_rate', config.TFT_PARAMS['learning_rate'])
        drp = self.params.get('dropout', 0.1)

        models = []
        models.append(
            TFT(
                h=config.TFT_PARAMS['h'],
                input_size=config.TFT_PARAMS['input_size'],

                # Parametry tréninku
                max_steps=config.TFT_PARAMS['max_steps'],
                early_stop_patience_steps=config.TFT_PARAMS['early_stop_patience_steps'],
                batch_size=config.TFT_PARAMS['batch_size'],

                # Architektura
                hidden_size=h_size,
                learning_rate=lr,
                dropout=drp,
                n_head=4,

                scaler_type=config.TFT_PARAMS['scaler_type'],
                loss=self.loss,
                futr_exog_list=config.FUTR_EXOG_LIST,
                alias='TFT_Model',

                # Konfigurace GPU musí být zde
                **config.TRAINER_KWARGS
            )
        )
        return models

    def train(self, df_sales, df_guests):

        # --- KONTROLA A VYNUCENÍ GPU ARCHITEKTURY ---
        # Tato sekce řeší chybu "no kernel image is available" pro novou řadu 50xx.
        if config.TRAINER_KWARGS.get('accelerator') == 'gpu' and torch.cuda.is_available():
            # Použijeme architekturu sm_89 (RTX 40xx), která je kompatibilní s řadou 50xx a je PyTorchi známá.
            if not os.environ.get('TORCH_CUDA_ARCH_LIST'):
                os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"
                print("Vynuceno TORCH_CUDA_ARCH_LIST=8.9 pro řešení konfliktu binárního kódu.")

        print("\nINFO: Startuji trénink modelu (GPU Powered 🚀)...")

        # Nastavení validační sady pro Early Stopping. 168 hodin = 7 dní.
        val_size = 168

        # Sales
        self.nf_sales = NeuralForecast(
            models=self._build_models_list(),
            freq=config.FREQ
        )
        self.nf_sales.fit(df=df_sales, val_size=val_size)

        # Guests
        self.nf_guests = NeuralForecast(
            models=self._build_models_list(),
            freq=config.FREQ
        )
        self.nf_guests.fit(df=df_guests, val_size=val_size)

        print("INFO: Trénink dokončen.")

    def predict(self, future_df, S, tags):
        print(f"INFO: Generuji predikci...")
        # Predikce
        p_sales = self.nf_sales.predict(futr_df=future_df)
        p_guests = self.nf_guests.predict(futr_df=future_df)

        # Přejmenování
        p_sales = self._rename_output_columns(p_sales)
        p_guests = self._rename_output_columns(p_guests)

        # Rekonciliace
        final_sales = self._reconcile_detailed(p_sales, S, tags)
        final_guests = self._reconcile_detailed(p_guests, S, tags)

        return final_sales, final_guests

    def save_model(self, path="model_checkpoints"):
        print(f"INFO: Ukládám modely do '{path}'...")
        os.makedirs(path, exist_ok=True)
        if self.nf_sales:
            self.nf_sales.save(path=f"{path}/sales", model_index=None, overwrite=True)
        if self.nf_guests:
            self.nf_guests.save(path=f"{path}/guests", model_index=None, overwrite=True)

    def load_model(self, path="model_checkpoints"):
        print(f"INFO: Načítám modely z '{path}'...")
        try:
            self.nf_sales = NeuralForecast.load(path=f"{path}/sales")
            self.nf_guests = NeuralForecast.load(path=f"{path}/guests")
            return True
        except Exception as e:
            print(f"VAROVÁNÍ: Load selhal ({e}).")
            return False

    def _rename_output_columns(self, df):
        cols = df.columns
        try:
            col_median = [c for c in cols if 'median' in c or '0.5' in c][0]
            col_lo = [c for c in cols if 'lo' in c or '0.1' in c][0]
            col_hi = [c for c in cols if 'hi' in c or '0.9' in c][0]

            df = df.rename(columns={
                col_median: 'Forecast_Value',
                col_lo: 'y_pred_low',
                col_hi: 'y_pred_high'
            })
        except IndexError:
            pass
        return df

    def _reconcile_detailed(self, preds_df, S, tags):
        preds_clean = preds_df.reset_index()
        cols_to_process = ['Forecast_Value', 'y_pred_low', 'y_pred_high']
        final_df = preds_clean[['unique_id', 'ds']].copy()

        for col in cols_to_process:
            if col not in preds_clean.columns: continue
            temp_hat = preds_clean[['unique_id', 'ds', col]].rename(columns={col: 'yhat'})
            temp_hat = temp_hat.set_index('unique_id')
            try:
                rec = self.reconciler.reconcile(Y_hat_df=temp_hat, S=S, tags=tags)
                out_col = [c for c in rec.columns if 'BottomUp' in c][-1]
                rec = rec.reset_index().rename(columns={out_col: col})
                final_df = pd.merge(final_df, rec[['unique_id', 'ds', col]], on=['unique_id', 'ds'], how='left')
            except Exception:
                final_df = pd.merge(final_df, preds_clean[['unique_id', 'ds', col]], on=['unique_id', 'ds'], how='left')
        return final_df