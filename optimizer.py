import optuna
import pandas as pd
import multiprocessing
import logging
import importlib
import math
import config

# Potlačení logů
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelOptimizer:
    def __init__(self, train_df, horizon=24, n_trials=10):
        self.train_df = train_df
        self.horizon = horizon
        self.n_trials = n_trials

    def objective(self, trial):
        # Importujeme worker až TADY a vynutíme reload (pro Windows multiprocessing)
        import worker
        importlib.reload(worker)

        # 1. Návrh parametrů
        params = {
            # [FIX] Learning rate - bezpečnější rozsah
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),

            # Fixní parametry z Configu
            "hidden_size": config.TFT_PARAMS['hidden_size'],
            "batch_size": config.TFT_PARAMS['batch_size'],
            "attn_head_size": config.TFT_PARAMS['attn_head_size'],
            "input_size": config.TFT_PARAMS['input_size'],
            "max_steps": config.TFT_PARAMS['max_steps']
        }

        # 2. Serializace dat
        train_data_dict = self.train_df.to_dict(orient='list')

        # 3. Spuštění workeru (Izolovaný proces)
        queue = multiprocessing.Queue()
        ctx = multiprocessing.get_context('spawn') # Nutné pro Windows

        process = ctx.Process(
            target=worker.train_process_worker,
            args=(params, train_data_dict, self.horizon, queue)
        )

        process.start()
        process.join() # Čekáme na dokončení

        # 4. Získání výsledku
        # [FIX] Penalizace nastavena na 1 miliardu, aby ji Optuna ignorovala
        penalty_score = 1_000_000_000.0

        if not queue.empty():
            result = queue.get()

            # --- DEBUG VÝPISY DO KONZOLE ---
            if result['status'] == 'error':
                print(f"❌ CHYBA WORKERU (Trial {trial.number}): {result['message']}")
                return penalty_score
            # -------------------------------

            if result['status'] == 'ok':
                mae = result['mae']

                # [FIX] Ochrana proti NaN / Inf
                if mae is None or math.isnan(mae) or math.isinf(mae):
                    print(f"⚠️ VAROVÁNÍ (Trial {trial.number}): Model vrátil NaN/Inf.")
                    return penalty_score

                # [FIX] Ochrana proti explozi gradientů
                if mae > penalty_score:
                    print(f"⚠️ VAROVÁNÍ (Trial {trial.number}): MAE je příliš vysoké ({mae}).")
                    return penalty_score

                print(f"✅ OK (Trial {trial.number}): MAE = {mae:.4f}")
                return float(mae)
            else:
                return penalty_score
        else:
            print(f"❌ KRITICKÁ CHYBA (Trial {trial.number}): Worker neodpověděl (Crash/SegFault).")
            return penalty_score

    def optimize(self, streamlit_callback=None):
        print(f"INFO: Spouštím optimalizaci (Optuna)...")

        callbacks = []
        if streamlit_callback:
            callbacks.append(streamlit_callback)

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials, callbacks=callbacks)

        print(f"INFO: Nejlepší parametry: {study.best_params}")

        best = study.best_params.copy()
        best.update({
            'hidden_size': config.TFT_PARAMS['hidden_size'],
            'batch_size': config.TFT_PARAMS['batch_size'],
            'attn_head_size': config.TFT_PARAMS['attn_head_size']
        })
        return best