import pandas as pd
import numpy as np
import torch
import os
import gc
import config
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import HuberMQLoss
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp
import logging
import warnings

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# --- PERFORMANCE BOOST ---
torch.set_float32_matmul_precision('medium')

class ForecastModel:
    def __init__(self, best_params=None):
        self.nf_sales = None
        self.nf_guests = None
        self.reconciler = HierarchicalReconciliation(reconcilers=[BottomUp()])

        self.params = config.TFT_PARAMS.copy()
        if best_params:
            self.params.update(best_params)

        self.loss = HuberMQLoss(quantiles=[0.1, 0.5, 0.9], delta=1.0)
        self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"   [ForecastModel] GPU: {torch.cuda.get_device_name(0)} | Batch: {self.params['batch_size']} | Steps Limit: {self.params['max_steps']}")

    def _build_models_list(self):
        # --- FIX: Definice parametrů pro Trainer ---
        trainer_args = {
            'max_steps': self.params['max_steps'], # Limit kroků
            'accelerator': self.accelerator,
            'enable_model_summary': False,
            'enable_progress_bar': True,
            'check_val_every_n_epoch': 1
        }

        # --- FIX: Inicializace modelu BEZ trainer_kwargs v konstruktoru ---
        tft_model = TFT(
            h=self.params['h'],
            input_size=self.params['input_size'],

            # Parametry modelu
            hidden_size=self.params['hidden_size'],
            learning_rate=self.params['learning_rate'],
            scaler_type=self.params['scaler_type'],
            batch_size=self.params['batch_size'],
            dropout=self.params['dropout'],
            loss=self.loss,
            futr_exog_list=config.FUTR_EXOG_LIST,
            alias='TFT_Model',

            # Early Stopping (Model level)
            early_stop_patience_steps=self.params['early_stop_patience_steps']
        )

        # --- FIX: Ruční přiřazení trainer_kwargs ---
        tft_model.trainer_kwargs = trainer_args

        # Windows Fix
        tft_model.num_workers_loader = 0
        tft_model.drop_last_loader = False

        return [tft_model]

    def train(self, df_sales, df_guests):
        print(f"INFO: Startuji trénink (Sales). Limit kroků: {self.params['max_steps']}...")
        self.nf_sales = NeuralForecast(models=self._build_models_list(), freq=config.FREQ)
        self.nf_sales.fit(df=df_sales, val_size=self.params['h'])

        torch.cuda.empty_cache()

        print(f"INFO: Startuji trénink (Guests). Limit kroků: {self.params['max_steps']}...")
        self.nf_guests = NeuralForecast(models=self._build_models_list(), freq=config.FREQ)
        self.nf_guests.fit(df=df_guests, val_size=self.params['h'])

        torch.cuda.empty_cache()
        gc.collect()

    def predict(self, future_df, S, tags):
        print("INFO: Generuji predikce...")
        p_sales = self.nf_sales.predict(futr_df=future_df)
        torch.cuda.empty_cache()
        p_guests = self.nf_guests.predict(futr_df=future_df)
        torch.cuda.empty_cache()

        p_sales = self._rename_output_columns(p_sales)
        p_guests = self._rename_output_columns(p_guests)

        final_sales = self._reconcile_detailed(p_sales, S, tags)
        final_guests = self._reconcile_detailed(p_guests, S, tags)
        return final_sales, final_guests

    def _rename_output_columns(self, df):
        cols = df.columns
        col_median = next((c for c in cols if 'median' in c or '0.5' in c), 'TFT_Model')
        col_lo = next((c for c in cols if 'lo' in c or '0.1' in c), None)
        col_hi = next((c for c in cols if 'hi' in c or '0.9' in c), None)

        mapping = {col_median: 'Forecast_Value'}
        if col_lo: mapping[col_lo] = 'y_pred_low'
        if col_hi: mapping[col_hi] = 'y_pred_high'
        return df.rename(columns=mapping)

    def _reconcile_detailed(self, preds_df, S, tags):
        preds_clean = preds_df.reset_index()
        cols_to_process = ['Forecast_Value', 'y_pred_low', 'y_pred_high']
        final_df = preds_clean[['unique_id', 'ds']].copy()

        for col in cols_to_process:
            if col not in preds_clean.columns: continue
            temp_hat = preds_clean[['unique_id', 'ds', col]].rename(columns={col: 'yhat'}).set_index('unique_id')
            try:
                rec = self.reconciler.reconcile(Y_hat_df=temp_hat, S=S, tags=tags)
                out_col = [c for c in rec.columns if 'BottomUp' in c][-1]
                rec = rec.reset_index().rename(columns={out_col: col})
                final_df = pd.merge(final_df, rec[['unique_id', 'ds', col]], on=['unique_id', 'ds'], how='left')
            except:
                final_df = pd.merge(final_df, preds_clean[['unique_id', 'ds', col]], on=['unique_id', 'ds'], how='left')
        return final_df

    def save_model(self, path):
        if not os.path.exists(path): os.makedirs(path)
        self.nf_sales.save(path + "/sales", save_dataset=True, overwrite=True)
        self.nf_guests.save(path + "/guests", save_dataset=True, overwrite=True)
        print(f"INFO: Model uložen do {path}.")

    def load_model(self, path):
        try:
            self.nf_sales = NeuralForecast.load(path + "/sales")
            self.nf_guests = NeuralForecast.load(path + "/guests")
            return True
        except:
            return False