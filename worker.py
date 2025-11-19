import pandas as pd
import torch
import logging
import warnings
import os
import sys

# Nutné importy pro proces
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import HuberMQLoss

# Potlačení bordelu v logu
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

def train_process_worker(params, train_data_dict, horizon, queue):
    """
    Izolovaný proces pro trénink jednoho modelu v Optuně.
    """
    try:
        # 1. Rekonstrukce DataFrame z dict (rychlejší než pickle celého DF)
        train_df = pd.DataFrame(train_data_dict)

        # Ujistíme se, že 'ds' je datetime
        if 'ds' in train_df.columns:
            train_df['ds'] = pd.to_datetime(train_df['ds'])

        # 2. Definice Modelu s OCHRANOU PAMĚTI
        # Zde musíme mít stejné optimalizace jako v hlavním modelu!
        model = TFT(
            h=horizon,
            input_size=168, # Fixní podle configu
            hidden_size=params['hidden_size'],
            learning_rate=params['learning_rate'],
            dropout=params['dropout'],
            batch_size=params['batch_size'],
            max_steps=500,  # Pro optimalizaci stačí méně kroků
            loss=HuberMQLoss(quantiles=[0.1, 0.5, 0.9]),
            scaler_type='robust',
            alias='TFT_Optuna',

            # --- GPU OPTIMALIZACE (KRITICKÉ) ---
            accelerator="gpu",
            precision="16-mixed",  # <--- TOTO CHYBĚLO! (Šetří 50% VRAM)

            # --- WINDOWS FIX ---
            num_workers_loader=0,
            drop_last_loader=False,
            enable_model_summary=False,
            enable_progress_bar=False
        )

        # Manuální fix pro jistotu
        model.num_workers_loader = 0

        # 3. Trénink
        nf = NeuralForecast(models=[model], freq='h')

        # Zkusíme trénink. Pokud dojde paměť, zachytíme to.
        try:
            nf.fit(df=train_df)

            # Validace (posledních 24h z tréninkových dat jako proxy)
            # V reálu bys měl mít split, ale pro rychlost použijeme in-sample chybu posledního okna
            # Nebo lépe: cross_validation, ale to je pomalé.
            # Pro jednoduchost zde jen vrátíme náhodnou metriku nebo placeholder,
            # ideálně bys měl udělat nf.cross_validation().

            # Rychlý odhad chyby (Cross Validation na posledním okně)
            cv_df = nf.cross_validation(df=train_df, n_windows=1, step_size=horizon)
            mae = (cv_df['y'] - cv_df['TFT_Optuna']).abs().mean()

            queue.put({'status': 'ok', 'mae': mae})

        except RuntimeError as e:
            if "out of memory" in str(e):
                # Uvolnění paměti
                torch.cuda.empty_cache()
                queue.put({'status': 'error', 'message': 'OOM'})
            else:
                queue.put({'status': 'error', 'message': str(e)})

    except Exception as e:
        queue.put({'status': 'error', 'message': str(e)})
    finally:
        # Úklid GPU
        torch.cuda.empty_cache()