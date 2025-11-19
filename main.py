import pandas as pd
import sys
import config
from dataloader import DataLoader
from feature_engineer import FeatureEngineer
from weather_service import WeatherService
from forecast_model import ForecastModel
from optimizer import ModelOptimizer
from visualizer import Visualizer

def main():
    print("==========================================")
    print("   BURGER KING DOBŠICE - AI FORECAST 2.0")
    print("==========================================")

    # --- 1. DATA LOADING ---
    loader = DataLoader()
    sales_df, guests_df, lat, lon = loader.load_data()

    if sales_df.empty:
        print("CRITICAL ERROR: Žádná data.")
        sys.exit(1)

    S, tags = DataLoader.get_hierarchy_matrix(sales_df)

    # --- 2. FEATURE ENGINEERING ---
    ws = WeatherService()
    weather_df = ws.get_weather_data(lat, lon, config.TRAIN_START_DATE, config.FORECAST_END)

    fe = FeatureEngineer()
    print("INFO: Aplikuji Feature Engineering...")

    # Transformace
    sales_df_aug = fe.transform(sales_df, weather_df)
    guests_df_aug = fe.transform(guests_df, weather_df)

    # Přidání LAGS (pouze pro historická data a analýzu/optimalizaci)
    sales_df_aug = fe.create_lags(sales_df_aug)
    guests_df_aug = fe.create_lags(guests_df_aug)

    # Příprava Future Rámce
    print("INFO: Připravuji budoucí rámec...")
    future_dates = pd.date_range(start=config.FORECAST_START, end=config.FORECAST_END, freq='H')
    unique_ids = sales_df['unique_id'].unique()

    future_df = pd.DataFrame()
    for uid in unique_ids:
        tmp = pd.DataFrame({'ds': future_dates, 'unique_id': uid})
        future_df = pd.concat([future_df, tmp])

    future_df_aug = fe.transform(future_df, weather_df)
    # Pozn: Na future_df neděláme create_lags, model si bere historii interně.

    # --- 3. MODEL (LOAD vs TRAIN) ---

    # Cesta pro uložení modelu
    MODEL_PATH = "bk_model_checkpoints"

    # Zde si můžeš přepínat: True = Vždy přetrénovat, False = Zkusit načíst
    FORCE_RETRAIN = True

    # Chceš optimalizovat parametry přes Optunu? (Trvá to déle)
    USE_OPTIMIZATION = True

    model = ForecastModel()
    model_loaded = False

    if not FORCE_RETRAIN:
        model_loaded = model.load_model(path=MODEL_PATH)

    if not model_loaded:
        print("\n--- MODEL NENALEZEN NEBO VYŽÁDÁN RETRAIN ---")

        # Příprava trénovacích dat (oříznutí o budoucnost)
        train_sales = sales_df_aug[sales_df_aug['ds'] < config.FORECAST_START]
        train_guests = guests_df_aug[guests_df_aug['ds'] < config.FORECAST_START]

        best_params = None
        if USE_OPTIMIZATION:
            print("INFO: Spouštím optimalizaci (hledám nejlepší parametry)...")
            optimizer = ModelOptimizer(train_sales, horizon=24, n_trials=10) # 10 pokusů stačí pro test
            best_params = optimizer.optimize()

        # Reinicializace modelu s novými parametry
        model = ForecastModel(best_params=best_params)

        # Trénink
        model.train(train_sales, train_guests)

        # Uložení pro příště
        model.save_model(path=MODEL_PATH)
    else:
        print("\nINFO: Používám již natrénovaný model z disku (přeskakuji trénink).")

    # --- 4. PREDIKCE ---
    horizon = len(future_dates)
    preds_sales, preds_guests = model.predict(future_df_aug, S, tags)

    # --- 5. VIZUALIZACE (LOKÁLNÍ) ---
    print("\n=== Generuji grafy do složky 'plots/' ===")
    viz = Visualizer(output_dir="plots")

    # Data pro graf (Potřebujeme sloupec 'y' z historie a 'Forecast_Value' z predikce)
    hist_sales = sales_df[['ds', 'unique_id', 'y']]

    # Vygenerujeme graf pro Total
    viz.plot_forecast(hist_sales, preds_sales, unique_id='Total')

    # Vygenerujeme graf pro jednotlivé kanály (pokud jich není 100)
    for uid in unique_ids:
        if uid != 'Total':
            viz.plot_forecast(hist_sales, preds_sales, unique_id=uid)

    # --- 6. EXPORT DO EXCELU ---
    output = preds_sales.sort_values(['ds', 'unique_id'])
    filename = "forecast_final.xlsx"
    try:
        output.to_excel(filename, index=False)
        print(f"\nINFO: Data uložena do '{filename}'.")
    except:
        print(f"\nCHYBA: Nelze uložit do '{filename}'. Asi ho máš otevřený.")

    print("DONE")

if __name__ == "__main__":
    main()