"""Download daily historical weather from Open-Meteo Archive API."""
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = DATA_DIR / "bengaluru_daily_weather.csv"

LATITUDE = 12.9716
LONGITUDE = 77.5946
START_DATE = "2018-01-01"
# Archive data can have a short reporting delay; yesterday is a conservative end date.
END_DATE = (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2)).date().isoformat()

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "wind_speed_10m_max", "weather_code"
        ],
        "timezone": "Asia/Kolkata",
    }
    print(f"Downloading daily weather for Bengaluru: {START_DATE} to {END_DATE}...")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "daily" not in payload or not payload["daily"].get("time"):
        raise RuntimeError(f"No daily data returned. API response keys: {list(payload)}")
    df = pd.DataFrame(payload["daily"]).rename(columns={
        "time": "date",
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "temperature_2m_mean": "temp_mean",
        "precipitation_sum": "precipitation",
        "wind_speed_10m_max": "wind_max",
        "weather_code": "weather_code",
    })
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows to {OUT}")
    print(df.tail(3).to_string(index=False))

if __name__ == "__main__":
    main()
