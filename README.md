# WeatherCast ML — Next-Day Weather Prediction

A machine learning project that predicts **next-day maximum temperature and precipitation occurrence** using historical weather observations, feature engineering, and time-based model evaluation. Includes an interactive Streamlit dashboard for exploring weather forecasts and model performance.

## Project Overview

WeatherCast ML investigates whether recent weather observations and calendar-based features can improve next-day predictions compared with a simple persistence baseline.

**Key highlights**
- Built a reproducible data-fetching and machine learning pipeline.
- Engineered lagged and rolling-window features from historical weather data.
- Trained Random Forest models for temperature regression and precipitation classification.
- Evaluated performance using a chronological train/test split.
- Compared temperature predictions against a persistence baseline.
- Developed an interactive Streamlit dashboard with live weather information and an ML evaluation section.

## Tech Stack

- **Language:** Python
- **Machine Learning:** scikit-learn, Random Forest
- **Data Processing:** pandas, NumPy
- **Visualization:** Streamlit and Plotly
- **Weather Data:** Open-Meteo API
- **Development:** VS Code, Git, GitHub

## 🎯 Prediction Tasks

1. **Temperature prediction:** Estimate the next day's maximum temperature.
2. **Precipitation prediction:** Classify whether measurable precipitation will occur the following day.

The default historical dataset uses weather observations from Bengaluru, India.

## 📊 Model Evaluation

The following results were recorded from the project's evaluation run. Confirm that they match `outputs/metrics.json` before publishing them.

| Metric | Random Forest / Model Result |
|---|---:|
| Temperature MAE | 0.832°C |
| Temperature RMSE | 1.093°C |
| Rain prediction accuracy | 84.5% |
| Rain prediction precision | 81.3% |
| Rain prediction recall | 83.9% |
| Rain prediction F1-score | 82.6% |

### Temperature baseline comparison

| Metric | Persistence Baseline | Random Forest |
|---|---:|---:|
| MAE | 1.143°C | 0.832°C |
| RMSE | 1.493°C | 1.093°C |

The Random Forest model achieved lower temperature prediction errors than the persistence baseline in this evaluation run.

**Evaluation approach:** The latest 20% of usable observations are reserved for testing using a chronological split, rather than randomly shuffling time-series observations.

## Dashboard Features

- Search for a city and view current weather conditions.
- Explore the seven-day weather forecast and hourly trends.
- Switch between Celsius and Fahrenheit.
- View UV, rainfall, and weather insights.
- Inspect machine learning evaluation metrics.
- Compare model performance with the persistence baseline.
- Explore actual versus predicted temperature and test predictions.

The live dashboard forecast comes from Open-Meteo. The historical machine learning evaluation is a separate component.

## Project Structure

```text
WeatherCast-ML/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_NOTES.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── fetch_data.py
│   └── train.py
└── outputs/
    ├── metrics.json
    └── test_predictions.csv
```

The `data/` and `models/` directories are generated locally when the pipeline runs. Large datasets and trained model files are excluded from version control.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the project's dependencies.
- Git
- An internet connection to fetch historical data and retrieve live weather information.

### 1. Clone the repository

```bash
git clone https://github.com/Keerthiii21/Weather-Prediction-Model.git
cd Weather-Prediction-Model
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Fetch historical weather data

```bash
python src/fetch_data.py
```

This downloads historical observations from the Open-Meteo Archive API.

### 4. Train and evaluate the models

```bash
python src/train.py
```

The pipeline generates evaluation metrics, test predictions, and trained models in the configured output directories.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to explore the dashboard.

## Data Source

Historical weather data is obtained from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).

The default configuration uses Bengaluru coordinates:

- Latitude: 12.9716° N
- Longitude: 77.5946° E

Features include historical temperature, precipitation, wind speed, weather codes, lagged observations, rolling statistics, and calendar information, as implemented in the feature pipeline.

## Limitations

- This is an educational and portfolio project, not an operational weather-warning system.
- Model performance is evaluated on one location and may not generalize to other regions.
- Sudden weather changes cannot always be predicted from historical observations alone.
- Precipitation metrics should be interpreted alongside class distributions and the chosen classification threshold.
- Live forecasts and historical model predictions use different data sources and methods.

For weather-related safety decisions, consult official forecasts and alerts.

## Future Improvements

- Evaluate model generalization across multiple cities.
- Compare Random Forest with additional gradient-boosting models.
- Investigate forecast-model inputs for improved next-day predictions.
- Add prediction uncertainty and probability calibration analysis.
- Track experiment configurations and dataset versions.
- Deploy the dashboard for public access.

## Author

**Keerthi Chandrashekhar**

GitHub: [@Keerthiii21](https://github.com/Keerthiii21)

---

*Built as a machine learning portfolio project to explore weather forecasting, time-series feature engineering, and model evaluation.*
