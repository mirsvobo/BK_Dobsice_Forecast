import multiprocessing
import pandas as pd
import torch
import time
from pytorch_lightning.callbacks import Callback
import importlib
import config
import os

# [FIX] Wrapper s metodami pro Streamlit
class QueueContainerWrapper:
    def __init__(self, queue):
        self.queue = queue

    def container(self):
        return self

    def put(self, data):
        self.queue.put(data)

    def markdown(self, text, unsafe_allow_html=False):
        # Bezpečně převedeme na string
        self.queue.put({'type': 'status', 'msg': str(text).replace('#', '').strip()})

    def info(self, text):
        self.queue.put({'type': 'status', 'msg': f"INFO: {str(text)}"})

    def success(self, text):
        self.queue.put({'type': 'status', 'msg': f"✅ {str(text)}"})

    def error(self, text):
        self.queue.put({'type': 'error', 'msg': str(text)})

class QueueTrainingUI(Callback):
    def __init__(self, total_steps, container):
        super().__init__()
        self.total_steps = total_steps
        self.queue = container.queue
        self.start_time = None
        self.refresh_rate = 5

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.queue.put({'type': 'status', 'msg': '🚀 Startuji trénink...'})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get('loss')
        if loss is None: return

        current_step = trainer.global_step
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)

        if current_step % self.refresh_rate == 0 or current_step >= self.total_steps:
            elapsed = time.time() - self.start_time
            self.queue.put({
                'type': 'metrics',
                'step': int(current_step),
                'total': int(self.total_steps),
                'loss': float(loss_val),
                'time': float(elapsed)
            })

# [FIX] Funkce nyní přijímá CESTY K SOUBORŮM, ne data
def run_parallel_training(sales_csv_path, guests_csv_path, best_params, queue):
    try:
        from forecast_model import ForecastModel

        # 1. Načtení dat ze souborů
        if not os.path.exists(sales_csv_path) or not os.path.exists(guests_csv_path):
            queue.put({'type': 'error', 'msg': 'Chyba: Dočasné soubory s daty nebyly nalezeny.'})
            return

        df_sales = pd.read_csv(sales_csv_path)
        df_guests = pd.read_csv(guests_csv_path)

        # Konverze data (CSV ztratí info o datetime, musíme obnovit)
        if 'ds' in df_sales.columns: df_sales['ds'] = pd.to_datetime(df_sales['ds'])
        if 'ds' in df_guests.columns: df_guests['ds'] = pd.to_datetime(df_guests['ds'])

        # 2. Inicializace modelu
        model = ForecastModel(best_params=best_params)

        # 3. Wrapper
        fake_container = QueueContainerWrapper(queue)

        # 4. Trénink
        model.train(
            df_sales,
            df_guests,
            ui_callback_cls=QueueTrainingUI,
            ui_container=fake_container
        )

        # 5. Uložení
        model.save_model(config.MODEL_CHECKPOINT_DIR)

        queue.put({'type': 'done', 'msg': '✅ Trénink úspěšně dokončen.'})

    except Exception as e:
        queue.put({'type': 'error', 'msg': f"Critical Error: {str(e)}"})
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()