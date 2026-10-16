# WeatherCast ML — Next-Day Weather Prediction

A reproducible machine-learning project that predicts **next-day maximum temperature** and **whether measurable precipitation will occur** from recent weather observations. It uses real historical weather data from the Open-Meteo Archive API, temporal feature engineering, time-ordered evaluation, baseline comparisons, and a small Streamlit dashboard.

> **Scope:** This is a portfolio-ready project scaffold. Run the pipeline and report the metrics it actually produces. Do not claim a specific accuracy until you have run it.

## Research question
Can recent weather observations and calendar/seasonality features improve next-day predictions over simple baselines?

## Features
- Fetches daily historical weather for Bengaluru (coordinates configurable).
- Builds lag and rolling-window features without using future information.
- Predicts next-day maximum temperature (regression) and precipitation occurrence (classification).
- Uses a chronological train/test split to reduce leakage.
- Compares a persistence baseline against a Random Forest model.
- Reports MAE/RMSE for temperature and precision/recall/F1 for precipitation.
- Saves processed data, fitted models, metrics, and test predictions.
- Includes an optional Streamlit dashboard for inspection.

## Quick start (VS Code, ~10–15 minutes)

### 1. Open the project
Extract the ZIP, then in VS Code choose **File → Open Folder** and select `WeatherCast-ML`.

### 2. Create a virtual environment
Open **Terminal → New Terminal** and run:

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, use Command Prompt terminal:
```bat
.venv\Scripts\activate.bat
```

### 3. Fetch data
```bash
python src/fetch_data.py
```
This downloads daily historical observations from Open-Meteo. An internet connection is required. If the API is unavailable, retry later.

### 4. Train and evaluate
```bash
python src/train.py
```
The script writes results to `outputs/` and models to `models/`. Read `outputs/metrics.json` and use those real results in your portfolio.

### 5. Interactive dashboard
The Streamlit interface includes live current conditions, city search, °C/°F toggle, 7-day forecast, hourly trends, UV/rain insights, and an ML evaluation tab. The live forecast is served by Open-Meteo; the trained historical ML model is evaluated separately.

### 6. Optional dashboard
```bash
streamlit run app.py
```
Open the local URL printed in the terminal.

## Project structure
```text
WeatherCast-ML/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── fetch_data.py
│   ├── features.py
│   └── train.py
├── data/                 # downloaded CSV goes here
├── models/               # trained models are saved here
└── outputs/               # metrics and predictions are generated here
```

## Data source and target definition
Open-Meteo Archive API: https://open-meteo.com/en/docs/historical-weather-api  
The default configuration uses Bengaluru (12.9716° N, 77.5946° E) and daily maximum/minimum temperature, mean temperature, precipitation sum, wind speed, and weather code. For each date *t*, features use weather observations through *t*, and the target is the next day's maximum temperature and precipitation occurrence.

## Evaluation
A chronological split reserves the latest 20% of usable rows as the test set. This is more appropriate for time series than randomly shuffling observations. Metrics are reported against a persistence baseline (tomorrow's max temperature ≈ today's max temperature) and a Random Forest model.

## Limitations and responsible use
- This is an educational forecasting model, not an operational weather warning system.
- Historical observations alone cannot reliably predict sudden local changes; official forecasts should be used for decisions affecting safety.
- Model performance can vary by season and location. A single-location test is not evidence of generalization to other regions.
- Precipitation is imbalanced in many climates; interpret F1/recall alongside class counts.
- API data availability and definitions can change.

## Suggested next improvements
1. Add a second location and evaluate out-of-location generalization.
2. Compare HistGradientBoosting / XGBoost (if installed) with Random Forest.
3. Use forecast-model inputs, not only lagged observations, for operational next-day forecasts.
4. Add confidence intervals and calibration analysis.
5. Track experiment configurations and version the dataset.

## Portfolio description (use only after running)
**WeatherCast ML — Next-Day Weather Prediction**  
Built a reproducible ML pipeline using historical weather observations, lag/rolling features, and chronological validation to predict next-day maximum temperature and precipitation occurrence. Compared a persistence baseline with a Random Forest model and reported regression/classification metrics. *(Add actual metric values after running the project.)*
