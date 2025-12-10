import optuna
import pandas as pd
import multiprocessing
import logging
import importlib
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
        # To zajistí, že změny v kódu workeru se projeví i bez restartu Streamlitu
        import worker
        importlib.reload(worker)

        # 1. Návrh parametrů (Search Space)
        params = {
            # Proměnné (hledané) parametry
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),

            # Fixní parametry (Výkonnostní metriky z Configu)
            "hidden_size": config.TFT_PARAMS['hidden_size'],
            "batch_size": config.TFT_PARAMS['batch_size'],
            "attn_head_size": config.TFT_PARAMS['attn_head_size'],
            "input_size": config.TFT_PARAMS['input_size'],

            # Pro optimalizaci používáme stejný počet kroků jako v configu
            "max_steps": config.TFT_PARAMS['max_steps']
        }

        # 2. Serializace dat (DataFrame -> Dict) pro přenos do procesu
        train_data_dict = self.train_df.to_dict(orient='list')

        # 3. Spuštění workeru (Izolovaný proces kvůli memory leakům a stabilitě)
        queue = multiprocessing.Queue()

        # Windows vyžaduje 'spawn'
        ctx = multiprocessing.get_context('spawn')

        process = ctx.Process(
            target=worker.train_process_worker,
            args=(params, train_data_dict, self.horizon, queue)
        )

        process.start()
        process.join() # Čekáme na dokončení procesu

        # 4. Získání výsledku z fronty
        if not queue.empty():
            result = queue.get()
            if result['status'] == 'ok':
                return result['mae']
            else:
                # Pokud to spadne (např. OOM), vrátíme "nekonečno", aby to Optuna zahodila
                return float('inf')
        else:
            return float('inf')

    def optimize(self, streamlit_callback=None):
        """
        Spustí optimalizaci.

        Args:
            streamlit_callback: Instance OptunaStreamlitCallback pro vizualizaci.
        """
        print(f"INFO: Spouštím optimalizaci (Optuna)...")
        print(f"      Fixované parametry: BS={config.TFT_PARAMS['batch_size']}, Hidden={config.TFT_PARAMS['hidden_size']}")

        # Příprava callbacků pro Optunu
        callbacks = []
        if streamlit_callback:
            callbacks.append(streamlit_callback)

        study = optuna.create_study(direction="minimize")

        # Spuštění optimalizace (blokující operace)
        study.optimize(self.objective, n_trials=self.n_trials, callbacks=callbacks)

        print(f"INFO: Nejlepší parametry: {study.best_params}")

        # Do výsledku vrátíme i ty fixní parametry, aby je ForecastModel mohl použít
        best = study.best_params.copy()
        best.update({
            'hidden_size': config.TFT_PARAMS['hidden_size'],
            'batch_size': config.TFT_PARAMS['batch_size'],
            'attn_head_size': config.TFT_PARAMS['attn_head_size']
        })
        return best