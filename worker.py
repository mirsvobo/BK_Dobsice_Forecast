import pandas as pd
import numpy as np
import torch
import gc
import logging
import sys
import os

# Importy pro model
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import HuberMQLoss
import config

def train_process_worker(params, train_df_dict, horizon, result_queue):
    """
    Worker běžící v izolovaném procesu.
    Tento soubor NESMÍ importovat Streamlit ani dashboard.py!
    """
    # 1. Absolutní potlačení logů v podprocesu
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    # Potlačení stdout/stderr, pokud nechceme vidět bordel v konzoli
    # sys.stdout = open(os.devnull, 'w')

    # 2. Nastavení pro RTX 5070
    gpu_name = "Unknown GPU"
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
        gpu_name = torch.cuda.get_device_name(0)

    # Debug print (uvidíš v konzoli)
    print(f"   [GPU Worker] Start na {gpu_name} | Batch: {params['batch_size']} | Hidden: {params['hidden_size']}", flush=True)

    # 3. Rekonstrukce DataFrame
    try:
        train_df = pd.DataFrame(train_df_dict)
        # Ujistíme se, že časový sloupec je datetime
        train_df['ds'] = pd.to_datetime(train_df['ds'])
    except Exception as e:
        result_queue.put({'status': 'error', 'message': f"Data error: {e}"})
        return

    current_batch_size = params.get('batch_size', 64)
    current_hidden_size = params.get('hidden_size', 64)

    # --- Tréninková smyčka (Auto-Scaling) ---
    while current_batch_size >= 1:
        try:
            # Úklid
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Definice modelu
            models = [
                TFT(
                    h=horizon,
                    input_size=config.TFT_PARAMS['input_size'],
                    max_steps=300,
                    early_stop_patience_steps=3,

                    learning_rate=params['learning_rate'],
                    hidden_size=current_hidden_size,
                    n_head=4,
                    dropout=params['dropout'],

                    batch_size=current_batch_size,
                    loss=HuberMQLoss(quantiles=[0.5]),
                    futr_exog_list=config.FUTR_EXOG_LIST,
                    scaler_type='robust',
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    alias='TFT_Opt',

                    **config.TRAINER_KWARGS
                )
            ]

            nf = NeuralForecast(models=models, freq=config.FREQ)

            # Cross Validation
            cv_df = nf.cross_validation(
                df=train_df,
                n_windows=1,
                step_size=24
            )

            y_true = cv_df['y']
            col_pred = [c for c in cv_df.columns if 'TFT_Opt' in c][0]
            y_pred = cv_df[col_pred]
            mae = np.mean(np.abs(y_true - y_pred))

            # Úspěch
            result_queue.put({'status': 'ok', 'mae': mae, 'batch': current_batch_size})
            return

        except Exception as e:
            err_msg = str(e)
            # Detekce OOM chyb
            is_oom = "out of memory" in err_msg or "CUDA error" in err_msg or "cudnn" in err_msg.lower()

            if is_oom:
                print(f"   [GPU Worker] ⚠️ OOM při Batch {current_batch_size}. Snižuji...", flush=True)
                if current_batch_size > 1:
                    current_batch_size //= 2
                elif current_hidden_size > 32:
                    current_hidden_size = 32
                else:
                    result_queue.put({'status': 'error', 'message': f'OOM CRITICAL: {err_msg}'})
                    return

                # Úklid před dalším pokusem
                if 'nf' in locals(): del nf
                if 'models' in locals(): del models
                gc.collect()
                torch.cuda.empty_cache()
            else:
                result_queue.put({'status': 'error', 'message': f"Training error: {err_msg}"})
                return

    result_queue.put({'status': 'error', 'message': 'Loop finished unexpectedly.'})