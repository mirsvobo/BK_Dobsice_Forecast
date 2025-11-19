import optuna
import pandas as pd
import multiprocessing
import logging
import importlib

# Potlačení logů
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelOptimizer:
    def __init__(self, train_df, horizon=24, n_trials=10):
        self.train_df = train_df
        self.horizon = horizon
        self.n_trials = n_trials

    def objective(self, trial):
        # --- FIX PICKLING ERROR ---
        # Importujeme worker až TADY a vynutíme reload.
        # Tím zajistíme, že multiprocessing dostane správný odkaz na funkci.
        import worker
        importlib.reload(worker)

        # 1. Návrh parametrů
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "batch_size": 32 # Pro optimalizaci stačí menší batch
        }

        # 2. Serializace dat (DataFrame -> Dict) pro přenos do procesu
        train_data_dict = self.train_df.to_dict(orient='list')

        # 3. Spuštění workeru
        queue = multiprocessing.Queue()

        # Windows vyžaduje 'spawn'
        ctx = multiprocessing.get_context('spawn')

        # Použijeme worker.train_process_worker (z modulu, ne z importu nahoře)
        process = ctx.Process(
            target=worker.train_process_worker,
            args=(params, train_data_dict, self.horizon, queue)
        )

        process.start()
        process.join() # Čekáme na dokončení

        # 4. Výsledek
        if not queue.empty():
            result = queue.get()
            if result['status'] == 'ok':
                return result['mae']
            else:
                # Pokud to spadne (OOM), vrátíme "nekonečno"
                return float('inf')
        else:
            return float('inf')

    def optimize(self):
        print(f"INFO: Spouštím optimalizaci (Optuna)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)

        print(f"INFO: Nejlepší parametry: {study.best_params}")
        return study.best_params