import optuna
import pandas as pd
import multiprocessing
import logging

# --- KLÍČOVÁ ZMĚNA: Importujeme z worker.py ---
from worker import train_process_worker

# Potlačení logů pro hlavní proces
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelOptimizer:
    def __init__(self, train_df, horizon=24, n_trials=10):
        self.train_df = train_df
        self.horizon = horizon
        self.n_trials = n_trials

    def objective(self, trial):
        # 1. Návrh parametrů
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", [64]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "batch_size": 64
        }

        # 2. Data serializujeme do slovníku (aby prošla přes Pickle do procesu)
        train_data_dict = self.train_df.to_dict(orient='list')

        # 3. Spuštění externího workeru
        queue = multiprocessing.Queue()

        # 'spawn' je nutný pro Windows + CUDA
        ctx = multiprocessing.get_context('spawn')

        process = ctx.Process(
            target=train_process_worker,
            args=(params, train_data_dict, self.horizon, queue)
        )

        process.start()
        process.join() # Čekáme, dokud worker neskončí (a neuvolní GPU)

        # 4. Zpracování výsledku
        if not queue.empty():
            result = queue.get()
            if result['status'] == 'ok':
                return result['mae']
            else:
                # print(f"Trial failed: {result.get('message')}")
                return float('inf')
        else:
            return float('inf')

    def optimize(self):
        print(f"INFO: Spouštím optimalizaci (Optuna + Process Isolation)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        print(f"INFO: Nejlepší parametry: {study.best_params}")
        return study.best_params