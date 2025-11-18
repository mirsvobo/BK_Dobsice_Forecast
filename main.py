import pandas as pd
import sys
import config
from dataloader import DataLoader
from feature_engineer import FeatureEngineer
from weather_service import WeatherService
from forecast_model import ForecastModel
from optimizer import ModelOptimizer

def main():
    print("==========================================")
    print("   BURGER KING DOBŠICE - AI FORECAST 2.0")
    print("==========================================")

    # 1. Načtení dat
    loader = DataLoader()
    sales_df, guests_df, lat, lon = loader.load_data()

    if sales_df.empty:
        print("CRITICAL ERROR: Žádná data.")
        sys.exit(1)

    S, tags = DataLoader.get_hierarchy_matrix(sales_df)

    # 2. Počasí & Features
    ws = WeatherService()
    weather_df = ws.get_weather_data(lat, lon, config.TRAIN_START_DATE, config.FORECAST_END)

    fe = FeatureEngineer()

    # Transformace a přidání Features
    print("INFO: Aplikuji Feature Engineering...")
    sales_df_aug = fe.transform(sales_df, weather_df)
    guests_df_aug = fe.transform(guests_df, weather_df)

    # Přidání LAGŮ (Zpožděné proměnné) - volitelné, ale doporučené pro analýzu
    sales_df_aug = fe.create_lags(sales_df_aug)
    guests_df_aug = fe.create_lags(guests_df_aug)

    # 3. Future DataFrame (Hodinový)
    print("INFO: Připravuji budoucí rámec...")
    future_dates = pd.date_range(start=config.FORECAST_START, end=config.FORECAST_END, freq='H')
    unique_ids = sales_df['unique_id'].unique()

    future_df = pd.DataFrame()
    for uid in unique_ids:
        tmp = pd.DataFrame({'ds': future_dates, 'unique_id': uid})
        future_df = pd.concat([future_df, tmp])

    future_df_aug = fe.transform(future_df, weather_df)
    # Poznámka: create_lags na future_df nevytváříme, protože nemáme historii 'y'.
    # Model TFT si interně bere historii z trénovacích dat.

    # 4. Optimalizace a Trénink

    # PŘEPÍNAČ: Chceš optimalizovat parametry? (True = trvá déle, ale lepší výsledek)
    RUN_OPTIMIZATION = True

    best_params = None

    # Data pro trénink (vše před začátkem předpovědi)
    train_sales = sales_df_aug[sales_df_aug['ds'] < config.FORECAST_START]
    train_guests = guests_df_aug[guests_df_aug['ds'] < config.FORECAST_START]

    if RUN_OPTIMIZATION:
        print("\n--- SPOUŠTÍM OPTIMALIZACI MODELU ---")
        # Optimalizujeme jen na prodejích (Sales), parametry použijeme i pro hosty
        optimizer = ModelOptimizer(train_sales, horizon=len(future_dates), n_trials=5) # Zkus 10-20 pro lepší výsledek
        best_params = optimizer.optimize()

    # Inicializace modelu (s parametry nebo bez)
    model = ForecastModel(best_params=best_params)
    model.train(train_sales, train_guests)

    # 5. Predikce
    horizon = len(future_dates)
    preds_sales, preds_guests = model.predict(future_df_aug, S, tags)

    # 6. Výstup
    print("\n=== Finální Hodinová Prognóza (Ukázka: Total - Oběd) ===")
    output = preds_sales.sort_values(['ds', 'unique_id'])

    mask_demo = (output['unique_id'] == 'Total') & (output['ds'].dt.hour == 12)
    print(output[mask_demo][['ds', 'Forecast_Value', 'y_pred_low', 'y_pred_high']].head(5))

    filename = "forecast_hourly_optimized.xlsx"
    try:
        output.to_excel(filename, index=False)
        print(f"\nINFO: Uloženo do '{filename}'.")
    except:
        print("\nCHYBA: Zavři Excel soubor!")

    print("DONE")

if __name__ == "__main__":
    main()