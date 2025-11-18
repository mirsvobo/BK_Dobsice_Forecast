import optuna
import pandas as pd
import numpy as np
import gc  # Důležité pro čištění RAM
import torch  # Důležité pro čištění VRAM
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import HuberMQLoss
import config
import logging

# Omezení logů
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

class ModelOptimizer:
    """
    Hledá nejlepší hyperparametry pomocí Optuny.
    """
    def __init__(self, train_df, horizon=24, n_trials=10):
        self.train_df = train_df
        self.horizon = horizon
        self.n_trials = n_trials

    def objective(self, trial):
        # 1. Návrh parametrů
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128]) # 128 může být na hraně, ale zkusíme
        dropout = trial.suggest_float("dropout", 0.1, 0.4)

        # Dynamický batch size podle hidden_size (aby se to vešlo do paměti)
        # Pokud je model velký (128 hidden), dáme menší batch.
        if hidden_size == 128:
            batch_size = 32
        else:
            batch_size = 64

            # 2. Inicializace modelu
        # Poznámka: batch_size jsem snížil z 128 na 64/32 pro stabilitu
        models = [
            TFT(
                h=self.horizon,
                input_size=config.TFT_PARAMS['input_size'],
                max_steps=500,
                early_stop_patience_steps=5,

                learning_rate=learning_rate,
                hidden_size=hidden_size,
                n_head=4,
                dropout=dropout,

                batch_size=batch_size,
                loss=HuberMQLoss(quantiles=[0.5]),
                futr_exog_list=config.FUTR_EXOG_LIST,
                scaler_type='robust',
                enable_progress_bar=False, # Vypneme bar pro rychlost
                alias='TFT_Opt',

                **config.TRAINER_KWARGS
            )
        ]

        nf = None
        try:
            nf = NeuralForecast(
                models=models,
                freq=config.FREQ
            )

            # 3. Cross Validation
            cv_df = nf.cross_validation(
                df=self.train_df,
                n_windows=2,
                step_size=24
            )

            y_true = cv_df['y']
            col_pred = [c for c in cv_df.columns if 'TFT_Opt' in c][0]
            y_pred = cv_df[col_pred]
            mae = np.mean(np.abs(y_true - y_pred))

            return mae

        except Exception as e:
            print(f"Trial error (pruned): {e}")
            # Pokud to spadne na paměť, vrátíme nekonečnou chybu, aby Optuna zkusila jiné parametry
            return float('inf')

        finally:
            # --- KRITICKÁ ČÁST: ÚKLID GPU ---
            # Smažeme objekty z Pythonu
            del nf
            del models
            # Vynutíme Python Garbage Collector
            gc.collect()
            # Vynutíme vyprázdnění cache na GPU
            torch.cuda.empty_cache()

    def optimize(self):
        print(f"INFO: Spouštím optimalizaci (Optuna, {self.n_trials} pokusů, GPU accelerated)...")
        # Přidáme gc.collect i před začátkem
        gc.collect()
        torch.cuda.empty_cache()

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)

        print(f"INFO: Nejlepší parametry: {study.best_params}")
        return study.best_params