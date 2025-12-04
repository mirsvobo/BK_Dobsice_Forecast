import pandas as pd
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

        # 2. Definice Modelu
        # Přebíráme parametry z params (včetně fixních z configu)
        model = TFT(
            h=horizon,
            input_size=params.get('input_size', 120),

            hidden_size=params['hidden_size'],      # Fix: 128
            batch_size=params['batch_size'],        # Fix: 64
            attn_head_size=params.get('attn_head_size', 4), # Fix: 4

            learning_rate=params['learning_rate'],  # Optimalizováno
            dropout=params['dropout'],              # Optimalizováno

            max_steps=params.get('max_steps', 300), # Fix: 500 (z optimizeru)

            loss=HuberMQLoss(quantiles=[0.1, 0.5, 0.9]),
            scaler_type='robust',
            alias='TFT_Optuna',

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

        # Pojistka pro Trainer args (aby nedocházelo k chybě trainer_kwargs)
        model.trainer_kwargs = {
            'accelerator': 'gpu',
            'max_steps': params.get('max_steps', 300),
            'enable_model_summary': False,
            'enable_progress_bar': False
        }

        # 3. Trénink
        nf = NeuralForecast(models=[model], freq='D')

        try:
            # Fit
            nf.fit(df=train_df)

            # Rychlý odhad chyby
            cv_df = nf.cross_validation(df=train_df, n_windows=1, step_size=horizon)
            mae = (cv_df['y'] - cv_df['TFT_Optuna']).abs().mean()

            queue.put({'status': 'ok', 'mae': mae})

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                queue.put({'status': 'error', 'message': 'OOM'})
            else:
                queue.put({'status': 'error', 'message': str(e)})

    except Exception as e:
        queue.put({'status': 'error', 'message': str(e)})
    finally:
        torch.cuda.empty_cache()