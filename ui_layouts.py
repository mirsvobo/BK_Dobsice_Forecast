import time
import pandas as pd
import streamlit as st
import plotly.express as px
from pytorch_lightning.callbacks import Callback

# --- 1. VIZUALIZACE PRO FINÁLNÍ TRÉNINK (NeuralForecast) ---
class PLTrainingUI(Callback):
    """
    Callback pro PyTorch Lightning.
    Vizualizuje průběh tréninku v reálném čase přímo do Streamlit kontejneru.
    Zobrazuje: Progress bar, Metriky (Loss, Čas, ETA), Graf Loss.
    """
    def __init__(self, total_steps, container):
        super().__init__()
        self.total_steps = total_steps
        self.container = container

        # Inicializace UI elementů uvnitř poskytnutého kontejneru
        self.status_text = container.empty()
        self.progress_bar = container.progress(0)

        # Sloupce pro metriky
        self.metrics_col1, self.metrics_col2, self.metrics_col3, self.metrics_col4 = container.columns(4)

        # Placeholder pro graf
        self.chart_placeholder = container.empty()

        self.start_time = None
        self.losses = []
        self.steps = []

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.losses = []
        self.steps = []
        self.status_text.markdown("### 🚀 Inicializuji trénink neurální sítě...")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Získání hodnoty Loss (může být dict nebo tensor)
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        if loss is None:
            return

        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        current_step = trainer.global_step

        # Aktualizujeme UI každých 5 kroků nebo na konci (pro úsporu výkonu renderingu)
        if current_step % 5 == 0 or current_step >= self.total_steps:
            elapsed = time.time() - self.start_time

            # Výpočty rychlosti a ETA
            speed = current_step / elapsed if elapsed > 0 else 0
            remaining_steps = self.total_steps - current_step
            eta_seconds = remaining_steps / speed if speed > 0 else 0
            progress = min(current_step / self.total_steps, 1.0)

            # Formátování času
            elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%M:%S", time.gmtime(eta_seconds))

            # Uložení historie pro graf
            self.losses.append(loss_val)
            self.steps.append(current_step)

            # 1. Aktualizace Metrik
            self.metrics_col1.metric("Krok", f"{current_step}/{self.total_steps}")
            self.metrics_col2.metric("Loss (Chyba)", f"{loss_val:.4f}")
            self.metrics_col3.metric("Uplynulo", elapsed_str)
            self.metrics_col4.metric("ETA (Zbývá)", eta_str)

            # 2. Aktualizace Progress Baru
            self.progress_bar.progress(progress)

            # 3. Aktualizace Grafu (Line Chart)
            # Vytvoříme jednoduchý DataFrame pro st.line_chart
            df_chart = pd.DataFrame({'Training Loss': self.losses}, index=self.steps)
            self.chart_placeholder.line_chart(df_chart, height=250)

            # 4. Status text
            self.status_text.markdown(f"⚡ **Trénuji...** Rychlost: `{speed:.1f} kroků/s` | GPU: Aktivní")

# --- 2. VIZUALIZACE PRO OPTUNU (Hyperparameter Tuning) ---
class OptunaStreamlitCallback:
    """
    Callback pro Optunu.
    Volá se po dokončení každého 'trialu' (pokusu).
    Aktualizuje tabulku výsledků a graf vývoje chyby.
    """
    def __init__(self, container, total_trials):
        self.container = container
        self.total_trials = total_trials
        self.container.markdown("### 🧬 Optimalizace Parametrů (Optuna)")

        # Layout pro Optunu
        self.status = container.empty()
        self.prog_bar = container.progress(0)

        self.col1, self.col2 = container.columns([2, 1])
        with self.col1:
            st.caption("📋 Historie pokusů (Top 10)")
            self.table_placeholder = st.empty()
        with self.col2:
            st.caption("📈 Vývoj chyby (MAE)")
            self.chart_placeholder = st.empty()

    def __call__(self, study, trial):
        # 1. Progress Bar a Status
        current_trial_num = trial.number + 1
        prog_val = min(current_trial_num / self.total_trials, 1.0)
        self.prog_bar.progress(prog_val)

        best_val = study.best_value
        self.status.markdown(
            f"**Běží pokus:** `{current_trial_num}/{self.total_trials}` | "
            f"**Nejlepší nalezené MAE:** `{best_val:.4f}` 🏆"
        )

        # 2. Tabulka dat (DataFrame)
        df = study.trials_dataframe()

        # Přejmenování sloupců pro hezčí zobrazení
        cols_map = {
            'number': 'ID',
            'value': 'MAE (Chyba)',
            'params_learning_rate': 'Learning Rate',
            'params_dropout': 'Dropout',
            'duration': 'Trvání (s)',
            'state': 'Stav'
        }

        # Filtrujeme jen sloupce, které v DataFrame skutečně existují
        avail_cols = [c for c in cols_map.keys() if c in df.columns]
        df_show = df[avail_cols].rename(columns=cols_map)

        # Formátování času trvání
        if 'Trvání (s)' in df_show.columns:
            df_show['Trvání (s)'] = df_show['Trvání (s)'].dt.total_seconds().round(1)

        # Zobrazení tabulky (řazeno od nejnovějšího)
        self.table_placeholder.dataframe(
            df_show.sort_values('ID', ascending=False).head(10),
            use_container_width=True,
            hide_index=True
        )

        # 3. Graf historie optimalizace (Scatter plot)
        # Zobrazujeme jen dokončené pokusy
        valid_trials = df[df['state'] == 'COMPLETE']
        if not valid_trials.empty:
            fig = px.scatter(
                valid_trials,
                x='number',
                y='value',
                title='Konvergence Optimalizace',
                labels={'number': 'Číslo pokusu', 'value': 'MAE (Chyba)'},
                template='plotly_white'
            )
            # Zvýraznění bodů
            fig.update_traces(marker=dict(size=10, color='#D62300', line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))

            self.chart_placeholder.plotly_chart(fig, use_container_width=True)