"""Train and evaluate next-day weather models using a chronological holdout."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, accuracy_score,
    precision_score, recall_score, f1_score, classification_report
)
from features import make_supervised_frame

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bengaluru_daily_weather.csv"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

def main():
    if not DATA.exists():
        raise FileNotFoundError("Dataset not found. Run: python src/fetch_data.py")
    raw = pd.read_csv(DATA, parse_dates=["date"])
    frame = make_supervised_frame(raw)
    target_temp = "target_temp_max_next_day"
    target_rain = "target_rain_next_day"
    feature_cols = [c for c in frame.columns if c not in ["date", target_temp, target_rain]]
    split = int(len(frame) * 0.8)
    train, test = frame.iloc[:split], frame.iloc[split:]
    X_train, X_test = train[feature_cols], test[feature_cols]
    y_temp_train, y_temp_test = train[target_temp], test[target_temp]
    y_rain_train, y_rain_test = train[target_rain], test[target_rain]

    # Baseline: tomorrow's maximum temperature equals today's observed maximum.
    # Current-date observations are known when the next-day prediction is made.
    temp_model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features=0.8,
        random_state=42, n_jobs=-1
    )
    rain_model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=4, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )
    temp_model.fit(X_train, y_temp_train)
    rain_model.fit(X_train, y_rain_train)
    temp_pred = temp_model.predict(X_test)
    rain_pred = rain_model.predict(X_test)

    # Naive persistence: use max temperature from the feature date (lag_1 corresponds to previous calendar day
    # due to feature shift; this is an intentionally simple reference baseline).
    # Use the closest available prior maximum (temp_max_lag_1) and clearly label the baseline.
    baseline_temp = test["temp_max_lag_1"].to_numpy()
    metrics = {
        "dataset": "Open-Meteo Archive API — Bengaluru",
        "rows_raw": int(len(raw)),
        "rows_supervised": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "split": "chronological 80/20; latest 20% held out",
        "precipitation_threshold_mm": 1.0,
        "temperature_max_next_day": {
            "random_forest_mae_c": round(float(mean_absolute_error(y_temp_test, temp_pred)), 3),
            "random_forest_rmse_c": round(float(np.sqrt(mean_squared_error(y_temp_test, temp_pred))), 3),
            "persistence_mae_c": round(float(mean_absolute_error(y_temp_test, baseline_temp)), 3),
            "persistence_rmse_c": round(float(np.sqrt(mean_squared_error(y_temp_test, baseline_temp))), 3),
        },
        "precipitation_occurrence_next_day": {
            "random_forest_accuracy": round(float(accuracy_score(y_rain_test, rain_pred)), 3),
            "random_forest_precision": round(float(precision_score(y_rain_test, rain_pred, zero_division=0)), 3),
            "random_forest_recall": round(float(recall_score(y_rain_test, rain_pred, zero_division=0)), 3),
            "random_forest_f1": round(float(f1_score(y_rain_test, rain_pred, zero_division=0)), 3),
            "test_positive_count": int(y_rain_test.sum()),
            "test_rows": int(len(y_rain_test)),
            "classification_report": classification_report(y_rain_test, rain_pred, zero_division=0, output_dict=True),
        },
    }
    MODELS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    joblib.dump({"model": temp_model, "features": feature_cols}, MODELS / "temperature_model.joblib")
    joblib.dump({"model": rain_model, "features": feature_cols}, MODELS / "precipitation_model.joblib")
    predictions = pd.DataFrame({
        "date": test["date"].dt.strftime("%Y-%m-%d"),
        "actual_temp_max_next_day": y_temp_test.to_numpy(),
        "predicted_temp_max_next_day": temp_pred,
        "actual_rain_next_day": y_rain_test.to_numpy(),
        "predicted_rain_next_day": rain_pred,
    })
    predictions.to_csv(OUTPUTS / "test_predictions.csv", index=False)
    with open(OUTPUTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k:v for k,v in metrics.items() if k != "precipitation_occurrence_next_day" or not isinstance(v, dict)}, indent=2))
    print("\nTemperature metrics:", metrics["temperature_max_next_day"])
    print("\nPrecipitation metrics:", {k:v for k,v in metrics["precipitation_occurrence_next_day"].items() if k != "classification_report"})
    print(f"\nSaved metrics: {OUTPUTS / 'metrics.json'}")
    print(f"Saved predictions: {OUTPUTS / 'test_predictions.csv'}")
    print(f"Saved models in: {MODELS}")

if __name__ == "__main__":
    main()
