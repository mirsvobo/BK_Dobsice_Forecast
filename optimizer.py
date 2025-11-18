import optuna
import pandas as pd
import numpy as np
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
        # Návrh parametrů
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.4)

        # Model pro optimalizaci (Rychlejší varianta)
        models = [
            TFT(
                h=self.horizon,
                input_size=config.TFT_PARAMS['input_size'],
                max_steps=500, # Méně kroků pro optunu
                early_stop_patience_steps=5, # Rychlé zastavení

                learning_rate=learning_rate,
                hidden_size=hidden_size,
                n_head=4,
                dropout=dropout,

                batch_size=128, # GPU to zvládne
                loss=HuberMQLoss(quantiles=[0.5]),
                futr_exog_list=config.FUTR_EXOG_LIST,
                scaler_type='robust',
                enable_progress_bar=False,
                alias='TFT_Opt',

                # GPU konfigurace (uvnitř modelu)
                **config.TRAINER_KWARGS
            )
        ]

        # Zde už GPU parametry nepředáváme
        nf = NeuralForecast(
            models=models,
            freq=config.FREQ
        )

        try:
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
            print(f"Trial error: {e}")
            return float('inf')

    def optimize(self):
        print(f"INFO: Spouštím optimalizaci (Optuna, {self.n_trials} pokusů, GPU accelerated)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        print(f"INFO: Nejlepší parametry: {study.best_params}")
        return study.best_params