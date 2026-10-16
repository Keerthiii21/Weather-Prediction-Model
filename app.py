"""WeatherCast ML — polished, mobile-first weather dashboard."""
from pathlib import Path
import json
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "outputs" / "metrics.json"
PREDICTIONS = ROOT / "outputs" / "test_predictions.csv"

st.set_page_config(page_title="WeatherCast", page_icon="🌤️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#17223b; --muted:#64748b; --sky:#eaf4ff; --glass:rgba(255,255,255,.72); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(155deg,#dcecff 0%,#f4f7ff 38%,#e9edff 100%); color:var(--ink); }
.block-container { max-width:1180px; padding-top:1.4rem; padding-bottom:3rem; }
#MainMenu, footer, header {visibility:hidden;}
h1,h2,h3 {font-family:'Manrope',sans-serif!important; color:#17223b!important; letter-spacing:-.035em;}
div[data-testid="stMetric"] {background:rgba(255,255,255,.67); border:1px solid rgba(255,255,255,.9); padding:15px 18px; border-radius:20px; box-shadow:0 8px 30px rgba(44,72,130,.06);}
div[data-testid="stMetricLabel"] {color:#64748b;}
div[data-testid="stMetricValue"] {color:#17223b;font-family:'Manrope',sans-serif;}
div[data-testid="stTabs"] button {font-weight:700;}
div[data-testid="stVerticalBlockBorderWrapper"] {background:rgba(255,255,255,.60); border:1px solid rgba(255,255,255,.82); border-radius:24px; padding:18px;}
div.stButton>button {border-radius:14px; border:0; background:#17223b; color:white; font-weight:700; padding:.65rem 1rem;}
div.stButton>button:hover {background:#334a78;color:white;}
.small-muted {color:#64748b;font-size:.85rem;}
.hero {background:linear-gradient(120deg,#4169b4,#7e9ff0 58%,#a8c9ff);color:white;border-radius:28px;padding:28px 30px;box-shadow:0 18px 55px rgba(54,91,165,.22);position:relative;overflow:hidden;}
.hero h1 {color:white!important;font-size:clamp(2.1rem,5vw,3.7rem);margin:0;line-height:1.05;}
.hero p {color:#edf4ff;font-size:1.03rem;margin:.6rem 0 0;}
.hero .temp {font-size:clamp(4rem,10vw,7rem);font-family:'Manrope';font-weight:800;letter-spacing:-.08em;line-height:1.05;}
.pill {display:inline-block;background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.28);padding:6px 11px;border-radius:999px;font-size:.78rem;margin-bottom:15px;}
.forecast-day {text-align:center;background:rgba(255,255,255,.74);border:1px solid white;border-radius:18px;padding:14px 8px;}
.forecast-day .emoji {font-size:1.8rem;}
.forecast-day .name {font-weight:700;font-size:.9rem;}
.forecast-day .range {font-size:.8rem;color:#64748b;margin-top:6px;}
@media(max-width:650px){.block-container{padding-left:1rem;padding-right:1rem}.hero{padding:22px 20px;border-radius:23px}}
</style>
""", unsafe_allow_html=True)

WEATHER_CODES = {
    0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"), 3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"), 48: ("Depositing rime fog", "🌫️"), 51: ("Light drizzle", "🌦️"),
    53: ("Drizzle", "🌦️"), 55: ("Dense drizzle", "🌧️"), 61: ("Slight rain", "🌦️"),
    63: ("Moderate rain", "🌧️"), 65: ("Heavy rain", "🌧️"), 71: ("Slight snow", "🌨️"),
    73: ("Snow", "🌨️"), 75: ("Heavy snow", "❄️"), 80: ("Rain showers", "🌦️"),
    81: ("Rain showers", "🌧️"), 82: ("Violent showers", "⛈️"), 95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with hail", "⛈️"), 99: ("Thunderstorm with hail", "⛈️")
}

def code_info(code):
    try: return WEATHER_CODES.get(int(code), ("Variable conditions", "🌤️"))
    except (TypeError, ValueError): return ("Variable conditions", "🌤️")

@st.cache_data(ttl=900, show_spinner=False)
def get_live_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature",
                    "is_day", "precipitation", "weather_code", "wind_speed_10m",
                    "wind_direction_10m"],
        "hourly": ["temperature_2m", "precipitation_probability", "precipitation",
                   "relative_humidity_2m", "wind_speed_10m", "weather_code", "uv_index"],
        "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min",
                  "precipitation_sum", "precipitation_probability_max", "uv_index_max",
                  "sunrise", "sunset", "wind_speed_10m_max"],
        "forecast_days": 7
    }
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def get_model_summary():
    if METRICS.exists():
        try: return json.loads(METRICS.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}

st.markdown('<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1.1rem;"><div style="font-weight:800;font-size:1.1rem;color:#17223b;">🌦️ WeatherCast <span style="font-weight:500;color:#64748b;">/ Forecast & Insights</span></div><div class="small-muted">LIVE WEATHER · ML LAB</div></div>', unsafe_allow_html=True)

with st.container(border=True):
    c1,c2,c3 = st.columns([1.6,1,1])
    with c1:
        city = st.text_input("Location", value=st.session_state.get("city","Bengaluru"), help="Enter a city, then click Update location.")
    with c2:
        units = st.selectbox("Temperature units", ["°C","°F"])
    with c3:
        st.write("")
        st.write("")
        update = st.button("Update location", width="stretch")
    if update:
        st.session_state["city"] = city.strip() or "Bengaluru"
        st.cache_data.clear()
        st.rerun()

city = st.session_state.get("city", "Bengaluru")
@st.cache_data(ttl=86400, show_spinner=False)
def geocode_city(name):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name":name,"count":1,"language":"en","format":"json"}, timeout=20)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results: return None
    return results[0]

try:
    with st.spinner(f"Getting live conditions for {city}…"):
        place = geocode_city(city)
        if not place: st.error(f"Couldn't find '{city}'. Try a larger city name."); st.stop()
        lat, lon = place["latitude"], place["longitude"]
        weather = get_live_weather(lat, lon)
except requests.RequestException as e:
    st.error("Live weather couldn't be loaded. Check your internet connection and try again.")
    st.caption(str(e))
    st.stop()

current = weather.get("current", {})
daily = weather.get("daily", {})
hourly = weather.get("hourly", {})
desc, emoji = code_info(current.get("weather_code"))
temp = current.get("temperature_2m", 0)
feels = current.get("apparent_temperature", temp)
if units == "°F":
    temp, feels = temp*9/5+32, feels*9/5+32
    high = [v*9/5+32 for v in daily.get("temperature_2m_max", [])]
    low = [v*9/5+32 for v in daily.get("temperature_2m_min", [])]
else:
    high, low = daily.get("temperature_2m_max", []), daily.get("temperature_2m_min", [])

st.markdown(f"""
<div class="hero">
  <div class="pill">● LIVE CONDITIONS &nbsp;·&nbsp; {place.get('country','')}</div>
  <div style="display:flex;justify-content:space-between;gap:20px;align-items:center;flex-wrap:wrap;">
    <div><div style="font-size:1.05rem;color:#eef4ff;">{place.get('name',city)}{', '+place.get('admin1','') if place.get('admin1') else ''}</div>
      <div class="temp">{temp:.0f}°</div><div style="font-size:1.12rem;font-weight:700;">{emoji} {desc}</div>
      <p>Feels like {feels:.0f}° · Local time {current.get('time','').replace('T',' ')}</p>
    </div>
    <div style="font-size:6rem;filter:drop-shadow(0 8px 16px #2d4f9980);">{emoji}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")
m1,m2,m3,m4 = st.columns(4)
m1.metric("Feels like", f"{feels:.0f}{units}")
m2.metric("Humidity", f"{current.get('relative_humidity_2m','—')}%")
m3.metric("Wind", f"{current.get('wind_speed_10m','—')} km/h")
m4.metric("Precipitation", f"{current.get('precipitation','—')} mm")

tabs = st.tabs(["7-day forecast", "Hourly", "Weather insights", "ML evaluation"])
with tabs[0]:
    st.subheader("The week ahead")
    st.caption("Seven-day forecast from the live weather API. Temperatures shown in selected units.")
    dates = daily.get("time", [])
    cols = st.columns(min(7, len(dates)) or 1)
    for i, d in enumerate(dates):
        ddesc, demoji = code_info(daily.get("weather_code", [None]*len(dates))[i])
        day_name = pd.to_datetime(d).strftime("%a")
        with cols[i]:
            st.markdown(f"""<div class="forecast-day"><div class="name">{'Today' if i==0 else day_name}</div><div class="emoji">{demoji}</div><div style="font-size:.76rem;color:#64748b;">{ddesc}</div><div style="font-weight:800;margin-top:7px;">{high[i]:.0f}°</div><div class="range">{low[i]:.0f}° / low</div><div class="range">☔ {daily.get('precipitation_probability_max',[0]*len(dates))[i]}%</div></div>""", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=high, name="High", mode="lines+markers", line=dict(width=3, color="#4778d0"), fill="tozeroy", fillcolor="rgba(71,120,208,.10)"))
    fig.add_trace(go.Scatter(x=dates, y=low, name="Low", mode="lines+markers", line=dict(width=3, color="#8a91c9")))
    fig.update_layout(height=300, margin=dict(l=8,r=8,t=25,b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12), yaxis_title=f"Temperature ({units})")
    st.plotly_chart(fig, width="stretch")
with tabs[1]:
    st.subheader("Next 24 hours")
    times = hourly.get("time", [])
    if times:
        now_time = current.get("time","")
        idx = next((i for i,t in enumerate(times) if t >= now_time), 0)
        end = min(idx+24, len(times))
        hdf = pd.DataFrame({
            "Time": pd.to_datetime(times[idx:end]),
            "Temperature (°C)": hourly.get("temperature_2m",[])[idx:end],
            "Rain chance (%)": hourly.get("precipitation_probability",[])[idx:end],
            "Wind (km/h)": hourly.get("wind_speed_10m",[])[idx:end],
            "UV index": hourly.get("uv_index",[None]*len(times))[idx:end],
        })
        if units == "°F": hdf["Temperature (°C)"] = hdf["Temperature (°C)"]*9/5+32; hdf.rename(columns={"Temperature (°C)":"Temperature (°F)"},inplace=True)
        fig = go.Figure()
        tempcol = [c for c in hdf.columns if c.startswith("Temperature")][0]
        fig.add_trace(go.Scatter(x=hdf["Time"], y=hdf[tempcol], name=tempcol, mode="lines+markers", line=dict(color="#4778d0",width=3)))
        fig.update_layout(height=310, margin=dict(l=8,r=8,t=20,b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,width="stretch")
        st.dataframe(hdf, width="stretch", hide_index=True)
with tabs[2]:
    st.subheader("Conditions & practical insights")
    uv = daily.get("uv_index_max", [None])[0]
    rain_chance = daily.get("precipitation_probability_max", [None])[0]
    sunrise = daily.get("sunrise", ["—"])[0]
    sunset = daily.get("sunset", ["—"])[0]
    windmax = daily.get("wind_speed_10m_max", ["—"])[0]
    i1,i2 = st.columns(2)
    with i1:
        st.metric("Max UV index today", uv if uv is not None else "—")
        st.metric("Rain probability today", f"{rain_chance}%" if rain_chance is not None else "—")
    with i2:
        st.metric("Sunrise", str(sunrise).split("T")[-1] if sunrise else "—")
        st.metric("Sunset", str(sunset).split("T")[-1] if sunset else "—")
    if uv is not None:
        if uv >= 8: st.warning("High UV: consider shade, protective clothing and sunscreen.")
        elif uv >= 6: st.info("Elevated UV: sun protection is a good idea around midday.")
        else: st.success("UV levels are lower than the high-risk range in this forecast.")
    if rain_chance is not None and rain_chance >= 60:
        st.info("Rain is likely according to the forecast. Consider carrying rain protection.")
    st.caption("These are rule-based summaries of the forecast, not medical or safety guarantees.")
with tabs[3]:
    st.subheader("Your trained ML model")
    st.caption(
        "This is separate from the live API forecast. The API forecast is the live weather view; "
        "the ML model is evaluated on historical held-out data."
    )
    if METRICS.exists():
        metrics = json.loads(METRICS.read_text(encoding="utf-8"))
        tm = metrics.get("temperature_max_next_day", {})
        rm = metrics.get("precipitation_occurrence_next_day", {})

        a, b, c = st.columns(3)
        a.metric("Test observations", metrics.get("test_rows", "—"))
        b.metric("Random Forest temperature MAE", f"{tm.get('random_forest_mae_c', '—')} °C")
        c.metric("Persistence baseline MAE", f"{tm.get('persistence_mae_c', '—')} °C")

        d, e, f = st.columns(3)
        d.metric("Rain F1", rm.get("random_forest_f1", "—"))
        e.metric("Rain precision", rm.get("random_forest_precision", "—"))
        f.metric("Rain recall", rm.get("random_forest_recall", "—"))

        st.markdown("#### Temperature model comparison")
        rf_mae = tm.get("random_forest_mae_c")
        baseline_mae = tm.get("persistence_mae_c")
        if isinstance(rf_mae, (int, float)) and isinstance(baseline_mae, (int, float)):
            comparison = go.Figure()
            comparison.add_trace(go.Bar(
                x=["Random Forest", "Persistence baseline"],
                y=[rf_mae, baseline_mae],
                text=[f"{rf_mae:.3f} °C", f"{baseline_mae:.3f} °C"],
                textposition="outside",
                marker_color=["#4778d0", "#a6b3cc"],
                hovertemplate="%{x}<br>MAE: %{y:.3f} °C<extra></extra>",
            ))
            comparison.update_layout(
                height=330,
                margin=dict(l=12, r=12, t=35, b=12),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Mean Absolute Error (°C)",
                yaxis=dict(rangemode="tozero"),
                showlegend=False,
            )
            st.plotly_chart(comparison, width="stretch")
            improvement = (baseline_mae - rf_mae) / baseline_mae * 100 if baseline_mae else 0
            st.caption(
                f"On this held-out test set, Random Forest reduced MAE by {improvement:.1f}% "
                "compared with the persistence baseline. Lower MAE is better."
            )
        else:
            st.info("Temperature comparison chart will appear when both MAE values are available in metrics.json.")

        if PREDICTIONS.exists():
            predictions = pd.read_csv(PREDICTIONS)
            st.markdown("#### Actual vs. predicted temperature")
            # Detect the target columns without requiring one exact naming convention.
            normalized = {col: str(col).lower().replace("-", "_").replace(" ", "_") for col in predictions.columns}
            actual_col = next(
                (col for col, name in normalized.items()
                 if "actual" in name and ("temp" in name or "temperature" in name)),
                None,
            )
            predicted_col = next(
                (col for col, name in normalized.items()
                 if ("predicted" in name or "prediction" in name or "pred" in name)
                 and ("temp" in name or "temperature" in name)),
                None,
            )

            if actual_col and predicted_col:
                chart_df = predictions[[actual_col, predicted_col]].copy()
                chart_df[actual_col] = pd.to_numeric(chart_df[actual_col], errors="coerce")
                chart_df[predicted_col] = pd.to_numeric(chart_df[predicted_col], errors="coerce")
                chart_df = chart_df.dropna().reset_index(drop=True)
                if not chart_df.empty:
                    chart_df["Test observation"] = range(1, len(chart_df) + 1)
                    actual_plot = go.Figure()
                    actual_plot.add_trace(go.Scatter(
                        x=chart_df["Test observation"], y=chart_df[actual_col],
                        mode="lines", name="Actual",
                        line=dict(color="#17223b", width=2),
                    ))
                    actual_plot.add_trace(go.Scatter(
                        x=chart_df["Test observation"], y=chart_df[predicted_col],
                        mode="lines", name="Predicted",
                        line=dict(color="#4778d0", width=2),
                    ))
                    actual_plot.update_layout(
                        height=350,
                        margin=dict(l=12, r=12, t=20, b=12),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Test observation",
                        yaxis_title="Maximum temperature (°C)",
                        legend=dict(orientation="h", y=1.12),
                    )
                    st.plotly_chart(actual_plot, width="stretch")
                    st.caption("The closer the two lines are, the closer the predictions are to the observed values.")
                else:
                    st.info("Temperature columns were found, but no numeric rows were available to chart.")
            else:
                st.info(
                    "Could not automatically identify the actual and predicted temperature columns. "
                    "Use the table below to inspect the available column names."
                )

            st.markdown("#### Inspect test predictions")
            st.caption("Use the column headers to sort. Choose how many held-out predictions to view.")
            row_limit = st.slider(
                "Rows to display", min_value=10, max_value=max(10, min(200, len(predictions))),
                value=min(50, max(10, len(predictions))), step=10,
                key="prediction_row_limit",
            )
            st.dataframe(predictions.head(row_limit), width="stretch", hide_index=True)
            st.download_button(
                "Download test predictions CSV",
                data=predictions.to_csv(index=False).encode("utf-8"),
                file_name="weathercast_test_predictions.csv",
                mime="text/csv",
            )
        else:
            st.info("The metrics are available, but test_predictions.csv was not found. Run `python src/train.py` again.")
    else:
        st.info("Train the ML models to unlock this section: run `python src/fetch_data.py`, then `python src/train.py` in the VS Code terminal.")
st.markdown('<p class="small-muted" style="text-align:center;margin-top:2rem;">WeatherCast ML · Live forecast powered by Open-Meteo · Educational project, not an official warning service.</p>', unsafe_allow_html=True)
