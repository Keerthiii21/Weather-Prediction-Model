"""Leakage-aware feature engineering for next-day targets."""
import pandas as pd

WEATHER_COLS = ["temp_max", "temp_min", "temp_mean", "precipitation", "wind_max", "weather_code"]

def make_supervised_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # Calendar features are known in advance.
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month_sin"] = __import__("numpy").sin(2 * __import__("numpy").pi * df["day_of_year"] / 365.25)
    df["month_cos"] = __import__("numpy").cos(2 * __import__("numpy").pi * df["day_of_year"] / 365.25)

    # Recent observations available at prediction time.
    for col in WEATHER_COLS:
        for lag in (1, 2, 3, 7, 14):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        df[f"{col}_rolling_mean_7"] = df[col].shift(1).rolling(7, min_periods=5).mean()

    # Predict the next day, using data known as of the current row.
    df["target_temp_max_next_day"] = df["temp_max"].shift(-1)
    df["target_rain_next_day"] = (df["precipitation"].shift(-1).fillna(0) >= 1.0).astype(int)

    # Current-day raw weather values are not used directly; all weather predictors are lagged.
    # Include observations from the feature date itself: at the end of date t, these are known
    # and can legitimately be used to predict date t+1.
    feature_cols = WEATHER_COLS + [c for c in df.columns if "_lag_" in c or "_rolling_mean_" in c]
    feature_cols += ["month", "day_of_year", "day_of_week", "month_sin", "month_cos"]
    out = df[["date", "target_temp_max_next_day", "target_rain_next_day"] + feature_cols].copy()
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return out
