import streamlit as st
import time
from pytorch_lightning.callbacks import Callback

class StreamlitTrainingCallback(Callback):
    def __init__(self, container, total_epochs, model_name="Model", y_std=1.0):
        self.container = container
        self.progress_bar = container.progress(0)
        self.status_text = container.empty()
        self.chart_place = container.empty()

        self.total_epochs = total_epochs
        self.model_name = model_name
        self.y_std = y_std

        self.losses = []
        self.last_update_time = 0

    def on_train_epoch_end(self, trainer, pl_module):
        current_loss = trainer.callback_metrics.get("train_loss")

        if current_loss is not None:
            loss_value = float(current_loss)
            real_error = loss_value * self.y_std
            self.losses.append(real_error)

            # --- THROTTLING (Zpomalení aktualizací UI) ---
            # Aktualizujeme UI maximálně každých 0.5 sekundy, aby nespadl Websocket
            current_time = time.time()
            if current_time - self.last_update_time > 0.5 or trainer.current_epoch == self.total_epochs - 1:
                self.last_update_time = current_time

                # Update progress
                current_epoch = trainer.current_epoch + 1
                progress = min(current_epoch / self.total_epochs, 1.0)
                self.progress_bar.progress(progress)

                # Update text
                unit = "Kč" if "Sales" in self.model_name or "Tržby" in self.model_name else "hostů"
                self.status_text.markdown(
                    f"**{self.model_name}** | Epocha {current_epoch} | "
                    f"Chyba: **{real_error:.0f} {unit}**"
                )

                # Update chart (jen pokud máme dost dat, ať to nebliká)
                if len(self.losses) > 5:
                    self.chart_place.line_chart(self.losses)