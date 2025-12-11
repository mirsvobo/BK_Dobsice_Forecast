import pandas as pd
import numpy as np
import torch
import logging
import warnings
import os

# --- 1. POTLAČENÍ VAROVÁNÍ ---
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# --- PERFORMANCE BOOST ---
torch.set_float32_matmul_precision('medium')

from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import HuberMQLoss

def train_process_worker(params, train_data_dict, horizon, queue):
    """
    Izolovaný proces pro trénink jednoho modelu v Optuně.
    """
    try:
        # 1. Rekonstrukce DataFrame
        train_df = pd.DataFrame(train_data_dict)
        if 'ds' in train_df.columns:
            train_df['ds'] = pd.to_datetime(train_df['ds'])

        # --- KONTROLA DAT ---
        # 1. Záporné hodnoty (Target)
        min_y = train_df['y'].min()
        if min_y < 0:
            queue.put({'status': 'error', 'message': f'CRITICAL: Data obsahují záporné hodnoty! Min: {min_y}. Vymažte cache (klávesa C).'})
            return

        # 2. NaN hodnoty (Kdekoliv)
        if train_df.isnull().values.any():
            queue.put({'status': 'error', 'message': 'Dataset obsahuje NaN hodnoty.'})
            return

        # 3. Nekonečné hodnoty (Inf)
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        if np.isinf(train_df[numeric_cols]).values.any():
            queue.put({'status': 'error', 'message': 'Dataset obsahuje nekonečné hodnoty (inf).'})
            return

        # 2. Definice Modelu
        model = TFT(
            h=horizon,
            input_size=params.get('input_size', 120),

            hidden_size=params['hidden_size'],
            batch_size=params['batch_size'],
            attn_head_size=params.get('attn_head_size', 4),

            learning_rate=params['learning_rate'],
            dropout=params['dropout'],

            max_steps=params.get('max_steps', 300),

            loss=HuberMQLoss(quantiles=[0.1, 0.5, 0.9]),
            scaler_type='robust',
            alias='TFT_Optuna', # I když zde dáme alias, ověříme si ho dynamicky níže

            # --- GPU OPTIMALIZACE ---
            accelerator="gpu",

            # --- WINDOWS FIX ---
            num_workers_loader=0,
            drop_last_loader=False,
            enable_model_summary=False,
            enable_progress_bar=False
        )

        # Pojistka pro Windows
        model.num_workers_loader = 0

        # Gradient Clipping (Stabilita)
        model.trainer_kwargs = {
            'accelerator': 'gpu',
            'max_steps': params.get('max_steps', 300),
            'gradient_clip_val': 0.5,
            'enable_model_summary': False,
            'enable_progress_bar': False
        }

        # 3. Trénink
        nf = NeuralForecast(models=[model], freq='D')

        try:
            # Fit
            nf.fit(df=train_df)

            # Cross Validation
            cv_df = nf.cross_validation(df=train_df, n_windows=1, step_size=horizon)

            # --- [FIX] DYNAMICKÁ DETEKCE SLOUPCE PREDIKCE ---
            # Hledáme sloupec, který není ID, datum, cutoff ani cílová proměnná.
            # To vyřeší problém, ať už se sloupec jmenuje 'TFT', 'TFT_Optuna' nebo jinak.
            excluded_cols = ['unique_id', 'ds', 'cutoff', 'y']
            pred_cols = [c for c in cv_df.columns if c not in excluded_cols]

            if not pred_cols:
                queue.put({'status': 'error', 'message': f'Nenalezen sloupec predikce. Dostupné: {list(cv_df.columns)}'})
                return

            # Vezmeme první nalezený modelový sloupec
            target_col = pred_cols[0]

            # Kontrola NaN v predikci
            if cv_df[target_col].isnull().any():
                queue.put({'status': 'error', 'message': f'Model ({target_col}) vrátil NaN predikce.'})
                return

            # Výpočet MAE
            mae = (cv_df['y'] - cv_df[target_col]).abs().mean()

            queue.put({'status': 'ok', 'mae': mae})

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                queue.put({'status': 'error', 'message': 'OOM (Out of Memory) - Snižte Batch Size'})
            else:
                queue.put({'status': 'error', 'message': str(e)})

    except Exception as e:
        queue.put({'status': 'error', 'message': str(e)})
    finally:
        torch.cuda.empty_cache()