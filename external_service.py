import pandas as pd
import numpy as np
import config

class TrafficService:
    """
    Generuje data o hustotě dopravy (Traffic Index).
    Pro Drive Thru je to klíčový faktor.
    """
    def __init__(self):
        pass

    def get_traffic_data(self, dates):
        print("INFO: Generuji data o dopravě (Traffic Index)...")

        df = pd.DataFrame({'ds': dates})
        df['ds'] = pd.to_datetime(df['ds'])
        df['hour'] = df['ds'].dt.hour
        df['dayofweek'] = df['ds'].dt.dayofweek

        # Simulace dopravy: 0 (prázdno) - 100 (zácpa)
        def simulate_traffic(row):
            h = row['hour']
            d = row['dayofweek']

            val = 10 # Noc

            if d < 5: # Pracovní dny
                if 7 <= h <= 9: val = 85 # Ranní špička
                elif 15 <= h <= 18: val = 95 # Odpolední špička (cesta z práce)
                elif 11 <= h <= 13: val = 60 # Oběd
                elif 6 <= h <= 22: val = 40 # Běžný provoz
            else: # Víkend
                if 10 <= h <= 16: val = 50 # Výlety
                elif 9 <= h <= 20: val = 30

            # Náhoda +/- 10
            val += np.random.randint(-5, 15)
            return max(0, min(100, val))

        df['traffic_index'] = df.apply(simulate_traffic, axis=1)
        return df[['ds', 'traffic_index']]