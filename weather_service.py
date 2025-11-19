import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import timedelta
import config

class WeatherService:
    """
    Chytrá služba pro počasí:
    1. Historie (Archive API) - pro trénování.
    2. Předpověď (Forecast API) - pro nejbližších 14 dní.
    3. Dlouhodobý průměr (Climatology) - pro vzdálenou budoucnost.
    """
    def __init__(self):
        # Cache oddělujeme verzí, abychom se vyhnuli konfliktům
        cache_name = config.WEATHER_CACHE_DIR + "_smart_v1"
        self.cache_session = requests_cache.CachedSession(cache_name, expire_after=3600*6) # Cache na 6 hodin
        self.retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=self.retry_session)

    def get_weather_data(self, lat, lon, start_date, end_date):
        print(f"INFO: Připravuji počasí pro {lat}, {lon}...")

        start_dt = pd.to_datetime(start_date)
        target_end_dt = pd.to_datetime(end_date)

        # Dnešek (bez času)
        today = pd.Timestamp.now().normalize()

        # Seznam datových rámců ke spojení
        frames = []

        # --- 1. FÁZE: HISTORIE (Do včerejška) ---
        history_end = min(target_end_dt, today - timedelta(days=1))
        if start_dt <= history_end:
            print("  -> [1/3] Stahuji historii (Archive API)...")
            df_hist = self._get_history_chunks(lat, lon, start_dt, history_end)
            if df_hist is not None:
                frames.append(df_hist)

        # --- 2. FÁZE: PŘEDPOVĚĎ (Dnes + 14 dní) ---
        # Open-Meteo forecast je spolehlivý cca 14-16 dní
        forecast_start = max(start_dt, today)
        forecast_end_limit = today + timedelta(days=14)
        actual_forecast_end = min(target_end_dt, forecast_end_limit)

        if forecast_start <= actual_forecast_end:
            print(f"  -> [2/3] Stahuji předpověď ({forecast_start.date()} - {actual_forecast_end.date()})...")
            df_forecast = self._fetch_api(
                "https://api.open-meteo.com/v1/forecast",
                lat, lon, forecast_start, actual_forecast_end
            )
            if df_forecast is not None:
                frames.append(df_forecast)

        # --- 3. FÁZE: KLIMATOLOGIE (Zbytek do budoucna) ---
        # Pokud chceme predikci dál než 14 dní, použijeme průměr z historie
        avg_start = actual_forecast_end + timedelta(hours=1)

        if avg_start < target_end_dt:
            print(f"  -> [3/3] Dopočítávám dlouhodobý průměr pro {avg_start.date()} až {target_end_dt.date()}...")

            # Potřebujeme historii pro výpočet průměrů (pokud ji už máme staženou, použijeme ji)
            if frames and not frames[0].empty:
                reference_df = frames[0] # Použijeme staženou historii
            else:
                # Fallback: Musíme stáhnout historii jen pro výpočet průměrů
                print("     (Stahuji referenční historii pro výpočet průměrů...)")
                ref_start = start_dt - timedelta(days=365*2) # 2 roky zpět
                reference_df = self._get_history_chunks(lat, lon, ref_start, today - timedelta(days=1))

            if reference_df is not None and not reference_df.empty:
                df_avg = self._calculate_climatology(reference_df, avg_start, target_end_dt)
                frames.append(df_avg)
            else:
                print("VAROVÁNÍ: Nemám z čeho spočítat průměr. Vyplňuji nulami.")

        # Spojení všeho dohromady
        if not frames:
            return pd.DataFrame()

        full_df = pd.concat(frames).drop_duplicates(subset=['ds']).sort_values('ds').reset_index(drop=True)

        # Ořezání přesně na požadovaný interval (pro jistotu)
        full_df = full_df[(full_df['ds'] >= start_dt) & (full_df['ds'] <= target_end_dt)]

        print(f"INFO: Počasí kompletní. Řádků: {len(full_df)}")
        return full_df

    def _get_history_chunks(self, lat, lon, start, end):
        """ Stahuje historii po rocích (aby nedošlo k chybě bufferu). """
        chunks = []
        curr = start
        while curr < end:
            nxt = min(curr + timedelta(days=365), end)
            # print(f"     - Chunk: {curr.date()} -> {nxt.date()}")
            try:
                df = self._fetch_api("https://archive-api.open-meteo.com/v1/archive", lat, lon, curr, nxt)
                if df is not None: chunks.append(df)
            except Exception as e:
                print(f"     - Chyba chunku: {e}")
            curr = nxt + timedelta(hours=1)

        return pd.concat(chunks) if chunks else None

    def _fetch_api(self, url, lat, lon, start, end):
        """ Obecná metoda pro volání API. """
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": config.WEATHER_PARAMS["hourly"],
            "timezone": config.WEATHER_PARAMS["timezone"]
        }
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]

            hourly = response.Hourly()
            start_ts = pd.to_datetime(hourly.Time(), unit="s", utc=True)
            end_ts = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
            interval = pd.Timedelta(seconds=hourly.Interval())

            dates = pd.date_range(start=start_ts, end=end_ts, freq=interval, inclusive='left')
            dates = dates.tz_convert(config.WEATHER_PARAMS["timezone"]).tz_localize(None)

            data = {"ds": dates}
            for i, var in enumerate(config.WEATHER_PARAMS["hourly"]):
                vals = hourly.Variables(i).ValuesAsNumpy()
                min_len = min(len(dates), len(vals))
                data[var] = vals[:min_len]

            data["ds"] = dates[:min_len]
            return pd.DataFrame(data)
        except Exception as e:
            # Ignorujeme chyby data range (když API nemá data pro daný den)
            return None

    def _calculate_climatology(self, history_df, start_date, end_date):
        """ Spočítá průměrné počasí pro daný měsíc, den a hodinu z historie. """
        df = history_df.copy()
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['hour'] = df['ds'].dt.hour

        # Agregace - průměr pro každý [měsíc, den, hodina]
        # Tím získáme "typický 1. leden v 10:00"
        avg_weather = df.groupby(['month', 'day', 'hour'])[config.WEATHER_PARAMS["hourly"]].mean().reset_index()

        # Vytvoření budoucího rámce
        future_dates = pd.date_range(start=start_date, end=end_date, freq='h')
        future_df = pd.DataFrame({'ds': future_dates})
        future_df['month'] = future_df['ds'].dt.month
        future_df['day'] = future_df['ds'].dt.day
        future_df['hour'] = future_df['ds'].dt.hour

        # Napojení průměrů
        result = pd.merge(future_df, avg_weather, on=['month', 'day', 'hour'], how='left')

        # Ošetření 29. února a chybějících dat (Forward Fill)
        result = result.fillna(method='ffill').fillna(0)

        return result.drop(columns=['month', 'day', 'hour'])