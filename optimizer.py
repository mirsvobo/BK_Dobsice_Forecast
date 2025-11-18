import optuna
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import HuberMQLoss
import config
import logging

# Potlačení logů Optuny, aby nezahlcovaly konzoli
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelOptimizer:
    """
    Hledá nejlepší hyperparametry pomocí Optuny a Cross-Validation.
    Testuje model na historických oknech, aby ověřil jeho robustnost.
    """
    def __init__(self, train_df, horizon=24, n_trials=10):
        self.train_df = train_df
        self.horizon = horizon
        self.n_trials = n_trials

    def objective(self, trial):
        """
        Jedna 'zkouška' (trial) Optuny.
        Navrhne parametry -> Natrénuje -> Otestuje (CrossVal) -> Vrátí chybu.
        """
        # 1. Definice prostoru hyperparametrů
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        # Attention heads (musí dělit hidden_size, pro zjednodušení fixujeme na 4)
        attn_heads = 4

        # 2. Sestavení modelu
        models = [
            TFT(
                h=self.horizon,
                input_size=config.TFT_PARAMS['input_size'],
                max_steps=300, # Méně kroků pro optimalizaci (pro rychlost)
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                n_head=attn_heads,
                dropout=dropout,
                loss=HuberMQLoss(quantiles=[0.5]), # Pro optimalizaci stačí medián
                futr_exog_list=config.FUTR_EXOG_LIST,
                scaler_type='robust',
                enable_progress_bar=False,
                alias='TFT_Opt'
            )
        ]

        nf = NeuralForecast(models=models, freq=config.FREQ)

        # 3. Cross-Validation (Validace v čase)
        # Otestujeme model na 2 oknech v minulosti (např. 2 různé dny)
        try:
            cv_df = nf.cross_validation(
                df=self.train_df,
                n_windows=2,      # Počet testovacích oken
                step_size=24      # Posun o 24h
            )

            # 4. Výpočet chyby (MAE - Mean Absolute Error)
            y_true = cv_df['y']
            # Sloupec s predikcí (může se jmenovat TFT_Opt nebo TFT_Opt-median)
            pred_col = [c for c in cv_df.columns if 'TFT_Opt' in c][0]
            y_pred = cv_df[pred_col]

            mae = np.mean(np.abs(y_true - y_pred))
            return mae

        except Exception as e:
            # Pokud pokus selže (např. memory error), vrátíme nekonečnou chybu
            return float('inf')

    def optimize(self):
        print(f"INFO: Spouštím optimalizaci (Optuna, {self.n_trials} pokusů)...")
        print("      (To může chvíli trvat, dej si kávu ☕)")

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)

        print("\n✅ INFO: Nejlepší parametry nalezeny:")
        print(study.best_params)
        return study.best_params