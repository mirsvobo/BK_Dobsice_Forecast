import sys
import pandas as pd
import json
import os
import time
import warnings
from pytorch_lightning.callbacks import Callback
import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from forecast_model import ForecastModel
import config

class DummyContainer:
    def container(self): return self
    def markdown(self, *args, **kwargs): pass
    def empty(self): return self
    def progress(self, *args): return self
    def columns(self, n): return [self]*n
    def metric(self, *args): pass
    def line_chart(self, *args, **kwargs): pass
    def write(self, *args): pass

# [NOVÉ] Vylepšený logger s výpočtem ETA a průměrné rychlosti
class JSONLoggerCallback(Callback):
    def __init__(self, total_steps, container=None):
        self.total_steps = total_steps
        self.start_time = None
        self.last_time = None
        self.log_every_n_steps = 10 # Posíláme častěji pro plynulost

        # Pro vyhlazení rychlosti (klouzavý průměr)
        self.recent_times = []

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.last_time = self.start_time
        print(json.dumps({"type": "status", "msg": "🚀 Inicializace GPU a tenzorů..."}), flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get('loss')
        if loss is None: return

        step = trainer.global_step
        current_time = time.time()

        # Aktualizujeme každých N kroků nebo na konci
        if step % self.log_every_n_steps == 0 or step >= self.total_steps:
            loss_val = loss.item() if hasattr(loss, 'item') else float(loss)

            # 1. Výpočet času od startu
            total_elapsed = current_time - self.start_time

            # 2. Výpočet rychlosti (kroky za sekundu) - vyhlazené
            step_time = current_time - self.last_time
            if step_time > 0:
                self.recent_times.append(step_time)
                if len(self.recent_times) > 10: self.recent_times.pop(0)
                avg_step_time = sum(self.recent_times) / len(self.recent_times)
                steps_per_sec = 1.0 / avg_step_time
            else:
                steps_per_sec = 0.0
                avg_step_time = 0.0

            # 3. Výpočet ETA (Remaining Time)
            remaining_steps = self.total_steps - step
            eta_seconds = remaining_steps * avg_step_time if avg_step_time > 0 else 0

            log_data = {
                "type": "metrics",
                "step": step,
                "total": self.total_steps,
                "loss": loss_val,
                "time": total_elapsed,
                "speed": steps_per_sec,      # kroky/s
                "avg_time": avg_step_time,   # s/krok
                "eta": eta_seconds           # zbývající čas v sekundách
            }
            print(json.dumps(log_data), flush=True)

        self.last_time = current_time

def run_training(sales_csv, guests_csv, params_json_path):
    try:
        with open(params_json_path, 'r') as f:
            best_params = json.load(f)

        df_sales = pd.read_csv(sales_csv)
        df_guests = pd.read_csv(guests_csv)

        if 'ds' in df_sales.columns: df_sales['ds'] = pd.to_datetime(df_sales['ds'])
        if 'ds' in df_guests.columns: df_guests['ds'] = pd.to_datetime(df_guests['ds'])

        print(json.dumps({"type": "status", "msg": "⚙️ Sestavuji model..."}), flush=True)

        model = ForecastModel(best_params=best_params)

        # Získáme informaci o max_steps z configu nebo params
        total_steps = best_params.get('max_steps', config.TFT_PARAMS['max_steps'])

        model.train(
            df_sales,
            df_guests,
            ui_callback_cls=JSONLoggerCallback, # Použijeme náš vylepšený logger
            ui_container=DummyContainer()
        )

        model.save_model(config.MODEL_CHECKPOINT_DIR)
        print(json.dumps({"type": "done", "msg": "✅ Trénink úspěšně dokončen."}), flush=True)

    except Exception as e:
        print(json.dumps({"type": "error", "msg": str(e)}), flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(1)
    run_training(sys.argv[1], sys.argv[2], sys.argv[3])