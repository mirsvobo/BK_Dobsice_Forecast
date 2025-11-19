import pandas as pd
import torch
import logging
import warnings
import os

# --- 1. POTLAČENÍ VAROVÁNÍ (SILENCE WARNINGS) ---
# Potlačení hlášky "No runtime found" od Streamlitu v podprocesech
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Potlačení Lightning a Torch varování
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# Nutné importy pro proces
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
        model = TFT(
            h=horizon,
            input_size=params.get('input_size', 60), # Fallback kdyby chybělo
            hidden_size=params['hidden_size'],
            learning_rate=params['learning_rate'],
            dropout=params['dropout'],
            batch_size=params['batch_size'],
            max_steps=300,  # Pro Optunu stačí málo kroků
            loss=HuberMQLoss(quantiles=[0.1, 0.5, 0.9]),
            scaler_type='robust',
            alias='TFT_Optuna',

            # --- GPU OPTIMALIZACE ---
            accelerator="gpu",
            precision="16-mixed",

            # --- WINDOWS FIX ---
            num_workers_loader=0,
            drop_last_loader=False,
            enable_model_summary=False,
            enable_progress_bar=False
        )

        # Pojistka pro Windows
        model.num_workers_loader = 0

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