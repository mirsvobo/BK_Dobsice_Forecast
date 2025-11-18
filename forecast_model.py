import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import HuberMQLoss
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp
import config
import warnings
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

class ForecastModel:
    """
    Řídí trénování a predikci pomocí modelu TFT.
    Podporuje dynamické hyperparametry z optimalizace.
    """
    def __init__(self, best_params=None):
        self.nf_sales = None
        self.nf_guests = None
        self.reconciler = HierarchicalReconciliation(reconcilers=[BottomUp()])

        # Robustní ztráta: Low(10%), Median(50%), High(90%)
        self.loss = HuberMQLoss(quantiles=[0.1, 0.5, 0.9], delta=1.0)

        # Pokud máme parametry z Optuny, použijeme je, jinak default
        self.params = best_params if best_params else config.TFT_PARAMS

    def _build_models_list(self):
        """
        Konfigurace TFT modelu.
        """
        # Bezpečné načtení parametrů (buď z Optuny, nebo z Configu)
        h_size = self.params.get('hidden_size', config.TFT_PARAMS['hidden_size'])
        lr = self.params.get('learning_rate', config.TFT_PARAMS['learning_rate'])
        drp = self.params.get('dropout', 0.1) # Default dropout

        models = []
        models.append(
            TFT(
                h=config.TFT_PARAMS['h'],
                input_size=config.TFT_PARAMS['input_size'],
                max_steps=config.TFT_PARAMS['max_steps'],

                # Dynamické parametry
                hidden_size=h_size,
                learning_rate=lr,
                dropout=drp,
                n_head=4, # Fixní pro stabilitu

                scaler_type=config.TFT_PARAMS['scaler_type'],
                loss=self.loss,
                futr_exog_list=config.FUTR_EXOG_LIST,
                alias='TFT_Model'
            )
        )
        return models

    def train(self, df_sales, df_guests):
        print("\nINFO: Trénuji finální model TFT...")

        # 1. Tržby
        print("  -> Trénink Tržby (Sales)...")
        self.nf_sales = NeuralForecast(models=self._build_models_list(), freq=config.FREQ)
        self.nf_sales.fit(df=df_sales)

        # 2. Hosté
        print("  -> Trénink Hosté (Guests)...")
        # Pro hosty použijeme stejné parametry (nebo bychom mohli optimalizovat zvlášť)
        self.nf_guests = NeuralForecast(models=self._build_models_list(), freq=config.FREQ)
        self.nf_guests.fit(df=df_guests)

        print("INFO: Trénování kompletní.")

    def predict(self, future_df, S, tags):
        print(f"INFO: Generuji predikci...")

        # future_df musí obsahovat features
        p_sales = self.nf_sales.predict(futr_df=future_df)
        p_guests = self.nf_guests.predict(futr_df=future_df)

        # Přejmenování a rekonciliace
        p_sales = self._rename_output_columns(p_sales)
        p_guests = self._rename_output_columns(p_guests)

        final_sales = self._reconcile_detailed(p_sales, S, tags)
        final_guests = self._reconcile_detailed(p_guests, S, tags)

        return final_sales, final_guests

    def _rename_output_columns(self, df):
        cols = df.columns
        # Hledáme sloupce podle klíčových slov
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
            # Fallback kdyby se názvy lišily
            print("VAROVÁNÍ: Nepodařilo se přejmenovat sloupce automaticky. Kontrola nutná.")

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