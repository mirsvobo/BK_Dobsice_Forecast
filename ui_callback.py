import time
import pandas as pd
import streamlit as st
from pytorch_lightning.callbacks import Callback

class StreamlitTrainingUI(Callback):
    """
    Callback, který v reálném čase vizualizuje trénink modelu NeuralForecast
    přímo do Streamlit aplikace (Progress bar, Loss Chart, Metriky, ETA).
    """
    def __init__(self, total_steps, chart_placeholder, metrics_placeholder, progress_bar, status_text):
        super().__init__()
        self.total_steps = total_steps
        self.chart_placeholder = chart_placeholder
        self.metrics_placeholder = metrics_placeholder
        self.progress_bar = progress_bar
        self.status_text = status_text

        self.start_time = None
        self.losses = []
        self.steps = []

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.losses = []
        self.steps = []
        self.status_text.write("🚀 Startuji trénink...")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Získání hodnoty Loss
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        if loss is None:
            return

        current_step = trainer.global_step

        # Aktualizujeme UI každých 5 kroků (aby to nebrzdilo výkon)
        if current_step % 5 == 0 or current_step == self.total_steps:
            current_time = time.time()
            elapsed = current_time - self.start_time

            # Výpočet rychlosti a ETA
            steps_per_sec = current_step / elapsed if elapsed > 0 else 0
            remaining_steps = self.total_steps - current_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

            elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%M:%S", time.gmtime(eta_seconds))

            loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
            self.losses.append(loss_val)
            self.steps.append(current_step)

            # 1. Graf Loss (chyby)
            df_chart = pd.DataFrame({'Training Loss': self.losses}, index=self.steps)
            self.chart_placeholder.line_chart(df_chart, height=250)

            # 2. Metriky (HTML)
            self.metrics_placeholder.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
                    <div style="text-align: center;">
                        <span style="font-size: 12px; color: gray;">KROK</span><br>
                        <strong style="font-size: 18px;">{current_step} / {self.total_steps}</strong>
                    </div>
                    <div style="text-align: center;">
                        <span style="font-size: 12px; color: gray;">LOSS</span><br>
                        <strong style="font-size: 18px; color: #D62300;">{loss_val:.4f}</strong>
                    </div>
                    <div style="text-align: center;">
                        <span style="font-size: 12px; color: gray;">ČAS</span><br>
                        <strong style="font-size: 18px;">{elapsed_str}</strong>
                    </div>
                    <div style="text-align: center;">
                        <span style="font-size: 12px; color: gray;">ZBÝVÁ (ETA)</span><br>
                        <strong style="font-size: 18px; color: green;">{eta_str}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 3. Progress Bar
            prog_val = min(current_step / self.total_steps, 1.0)
            self.progress_bar.progress(prog_val)
            self.status_text.write(f"⚡ Trénuji... Rychlost: {steps_per_sec:.2f} kroků/s")