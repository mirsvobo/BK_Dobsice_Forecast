import pandas as pd
import numpy as np
import torch
import os
import gc
import shutil
import config
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import HuberMQLoss
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp
import logging
import warnings

# Suppress warnings
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# --- GPU OPTIMIZATIONS ---
# Enables usage of Tensor Cores on Nvidia GPUs (Ampere/Volta/Turing)
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

        # Detect Hardware
        self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.device_id = [0] if torch.cuda.is_available() else "auto"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"   [ForecastModel] USING GPU: {torch.cuda.get_device_name(0)}")
            print(f"   [ForecastModel] Mixed Precision ENABLED (Speed Boost)")

    def _build_model(self):
        """
        Production-ready TFT configuration.
        """
        tft_model = TFT(
            h=self.params['h'],
            input_size=self.params['input_size'],
            hidden_size=self.params['hidden_size'],
            learning_rate=self.params['learning_rate'],
            scaler_type=self.params['scaler_type'],
            batch_size=self.params['batch_size'],
            dropout=self.params['dropout'],
            loss=self.loss,
            futr_exog_list=config.FUTR_EXOG_LIST,
            alias='TFT_Model',

            # --- PERFORMANCE TUNING ---
            max_steps=self.params['max_steps'],
            accelerator=self.accelerator,
            # '16-mixed' is crucial for RTX cards speedup
            precision='16-mixed' if self.accelerator == 'gpu' else '32',

            # Disable validation loop during training for pure speed
            limit_val_batches=0,
            num_sanity_val_steps=0,

            # Reduce logging overhead
            enable_model_summary=False,
            enable_progress_bar=True,
            enable_checkpointing=False,
            logger=False,

            # Windows DataLoader Safety (Must be 0 on Windows usually)
            num_workers_loader=0,
            random_seed=42
        )
        return tft_model

    def train(self, df_sales, df_guests):
        # Clean potential previous run garbage
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print("\n--- TRÉNINK MODELŮ (TFT) ---")

        # 1. SALES
        print(f" > Trénuji TRŽBY (Horizon: {self.params['h']} dní)...")
        model_sales = self._build_model()
        self.nf_sales = NeuralForecast(models=[model_sales], freq=config.FREQ)
        self.nf_sales.fit(df=df_sales)
        del model_sales

        # 2. GUESTS
        print(f" > Trénuji HOSTY (Horizon: {self.params['h']} dní)...")
        model_guests = self._build_model()
        self.nf_guests = NeuralForecast(models=[model_guests], freq=config.FREQ)
        self.nf_guests.fit(df=df_guests)
        del model_guests

        gc.collect()

    def predict(self, future_df, S, tags):
        print("INFO: Generuji predikce...")
        # Sales Prediction
        p_sales = self.nf_sales.predict(futr_df=future_df)
        p_sales = self._rename_output_columns(p_sales)

        # Guests Prediction
        p_guests = self.nf_guests.predict(futr_df=future_df)
        p_guests = self._rename_output_columns(p_guests)

        # Hierarchical Reconciliation
        print("INFO: Probíhá rekonsiliace (Bottom-Up)...")
        final_sales = self._reconcile_detailed(p_sales, S, tags)
        final_guests = self._reconcile_detailed(p_guests, S, tags)

        return final_sales, final_guests

    def _rename_output_columns(self, df):
        cols = df.columns
        # Locate quantiles dynamically
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

            # Prepare format for HierarchicalForecast
            temp_hat = preds_clean[['unique_id', 'ds', col]].rename(columns={col: 'yhat'}).set_index('unique_id')

            try:
                rec = self.reconciler.reconcile(Y_hat_df=temp_hat, S=S, tags=tags)
                # Find the reconciled column (usually ends with /BottomUp)
                out_col = [c for c in rec.columns if 'BottomUp' in c][-1]
                rec = rec.reset_index().rename(columns={out_col: col})

                final_df = pd.merge(final_df, rec[['unique_id', 'ds', col]], on=['unique_id', 'ds'], how='left')
            except Exception as e:
                # Fallback if reconciliation fails
                final_df = pd.merge(final_df, preds_clean[['unique_id', 'ds', col]], on=['unique_id', 'ds'], how='left')

        return final_df

    def save_model(self, path):
        if os.path.exists(path): shutil.rmtree(path)
        os.makedirs(path)
        self.nf_sales.save(path + "/sales", save_dataset=False, overwrite=True)
        self.nf_guests.save(path + "/guests", save_dataset=False, overwrite=True)

    def load_model(self, path):
        try:
            self.nf_sales = NeuralForecast.load(path + "/sales")
            self.nf_guests = NeuralForecast.load(path + "/guests")
            return True
        except:
            return False