
import os
import math
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Psilocybin Research Forecast",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONFIG
# ============================================================

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"

# Broad UK grid used for the research heatmap.
# This deliberately models environmental suitability at grid-cell level,
# rather than exposing individual occurrence locations.
GRID_STEP = 0.5

UK_BOUNDS = {
    "min_lat": 49.8,
    "max_lat": 59.0,
    "min_lon": -7.8,
    "max_lon": 1.8,
}

CACHE_TTL = 60 * 60

# ============================================================
# HELPERS
# ============================================================

def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, float(x)))


def normal_score(value, low, ideal_low, ideal_high, high):
    """Triangular/trapezoid ecological suitability score."""
    if value <= low or value >= high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        return 100.0 * (value - low) / (ideal_low - low)
    return 100.0 * (high - value) / (high - ideal_high)


def rain_score(mm_7d):
    # Transparent baseline, intentionally conservative.
    return clamp(normal_score(mm_7d, 3, 15, 65, 130))


def temperature_score(temp):
    return clamp(normal_score(temp, 2, 8, 15, 22))


def humidity_score(rh):
    return clamp(normal_score(rh, 45, 70, 95, 100))


def season_score(month):
    # Baseline phenology prior. This is not claimed to be a measured
    # probability; it is a starting prior that can later be calibrated.
    priors = {
        1: 5, 2: 3, 3: 2, 4: 2, 5: 3, 6: 5,
        7: 12, 8: 28, 9: 60, 10: 95, 11: 90, 12: 35
    }
    return priors.get(month, 0)


def habitat_score(habitat):
    # Broad land-cover prior. Replace/calibrate when licensed LCM data
    # is connected.
    h = str(habitat).lower()
    if "grass" in h or "heath" in h or "moor" in h or "pasture" in h:
        return 95
    if "wet" in h or "shrub" in h:
        return 65
    if "wood" in h or "forest" in h:
        return 35
    if "arable" in h or "crop" in h:
        return 20
    if "urban" in h or "built" in h:
        return 3
    return 45


def weighted_score(parts):
    weights = {
        "rain": 0.22,
        "temperature": 0.18,
        "humidity": 0.12,
        "season": 0.20,
        "habitat": 0.28,
    }
    return clamp(sum(parts[k] * weights[k] for k in weights))


def make_grid():
    lats = np.arange(UK_BOUNDS["min_lat"], UK_BOUNDS["max_lat"] + GRID_STEP, GRID_STEP)
    lons = np.arange(UK_BOUNDS["min_lon"], UK_BOUNDS["max_lon"] + GRID_STEP, GRID_STEP)
    return [(round(float(lat), 2), round(float(lon), 2)) for lat in lats for lon in lons]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_weather(lat, lon, forecast_days=7):
    params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": forecast_days,
        "timezone": "Europe/London",
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "soil_moisture_0_to_7cm",
            "soil_temperature_0_to_7cm",
            "wind_speed_10m",
        ]),
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "precipitation_sum",
            "rain_sum",
            "precipitation_probability_max",
        ]),
    }
    r = requests.get(OPEN_METEO_FORECAST, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def daily_weather_frame(data):
    d = pd.DataFrame(data["daily"])
    d["date"] = pd.to_datetime(d["time"]).dt.date
    return d


def calculate_forecast(data, habitat="Grassland"):
    daily = daily_weather_frame(data)

    # Approximate rolling rain from the forecast series. For production,
    # this should be combined with historical observed/reanalysis data.
    daily["rain_3d"] = daily["rain_sum"].rolling(3, min_periods=1).sum()
    daily["rain_7d"] = daily["rain_sum"].rolling(7, min_periods=1).sum()

    rows = []
    for _, r in daily.iterrows():
        month = r["date"].month
        parts = {
            "rain": rain_score(r["rain_7d"]),
            "temperature": temperature_score(r["temperature_2m_mean"]),
            "humidity": 75.0,  # replaced below by daily mean humidity when available
            "season": season_score(month),
            "habitat": habitat_score(habitat),
        }
        score = weighted_score(parts)
        rows.append({
            "date": r["date"],
            "score": round(score, 1),
            "rain_7d_mm": round(float(r["rain_7d"]), 1),
            "rain_mm": round(float(r["rain_sum"]), 1),
            "temperature": round(float(r["temperature_2m_mean"]), 1),
            "rain_probability": int(r.get("precipitation_probability_max", 0) or 0),
            "season_score": parts["season"],
            "habitat_score": parts["habitat"],
        })
    return pd.DataFrame(rows)


def build_research_map(forecast_date_offset=0):
    """Builds a broad environmental suitability map.

    The map is deliberately grid-based and does not expose individual
    occurrence points.
    """
    points = make_grid()
    rows = []

    target = date.today() + timedelta(days=forecast_date_offset)

    # Sampling the full UK grid live can create many API calls.
    # The production version should batch/cache weather by grid cell.
    # For the starter version, use a coarser sampling.
    sampled = points[::4]

    progress = st.progress(0, text="Building environmental forecast…")
    for i, (lat, lon) in enumerate(sampled):
        try:
            weather = get_weather(lat, lon, 7)
            df = calculate_forecast(weather)
            match = df[df["date"] == target]
            if match.empty:
                match = df.iloc[[min(forecast_date_offset, len(df)-1)]]
            r = match.iloc[0]
            rows.append({
                "lat": lat,
                "lon": lon,
                "score": float(r["score"]),
                "rain_7d": float(r["rain_7d_mm"]),
                "temp": float(r["temperature"]),
            })
        except Exception:
            pass
        progress.progress((i + 1) / max(1, len(sampled)))
    progress.empty()
    return pd.DataFrame(rows)


def score_label(score):
    if score >= 80:
        return "Very high environmental suitability"
    if score >= 65:
        return "High environmental suitability"
    if score >= 45:
        return "Moderate environmental suitability"
    if score >= 25:
        return "Low environmental suitability"
    return "Very low environmental suitability"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🍄 Research Forecast")
st.sidebar.caption("Psilocybe semilanceata environmental modelling")

tab_choice = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Live Map",
        "7-Day Forecast",
        "Location Explorer",
        "Research",
        "Other Fungi",
        "Model",
        "Guide",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "This research tool models broad environmental suitability. "
    "It does not verify the presence of individual mushrooms or provide "
    "collection instructions."
)

# ============================================================
# DASHBOARD
# ============================================================

if tab_choice == "Dashboard":
    st.title("🍄 Psilocybe semilanceata Research Forecast")
    st.write(
        "A UK environmental forecasting dashboard combining weather, "
        "seasonality, habitat and later-stage occurrence data."
    )

    col1, col2, col3, col4 = st.columns(4)
    today = date.today()

    with col1:
        st.metric("Forecast horizon", "7 days")
    with col2:
        st.metric("Model type", "Ecological baseline")
    with col3:
        st.metric("Primary species", "P. semilanceata")
    with col4:
        st.metric("Spatial mode", "Grid cells")

    st.subheader("Today's regional example")

    city = st.selectbox(
        "Choose a reference location",
        ["Manchester", "Liverpool", "Leeds", "Sheffield", "Birmingham",
         "Bristol", "Cardiff", "Edinburgh", "Glasgow", "London"],
    )

    coords = {
        "Manchester": (53.48, -2.24),
        "Liverpool": (53.41, -2.99),
        "Leeds": (53.80, -1.55),
        "Sheffield": (53.38, -1.47),
        "Birmingham": (52.49, -1.89),
        "Bristol": (51.45, -2.59),
        "Cardiff": (51.48, -3.18),
        "Edinburgh": (55.95, -3.19),
        "Glasgow": (55.86, -4.25),
        "London": (51.51, -0.13),
    }

    lat, lon = coords[city]
    try:
        data = get_weather(lat, lon)
        fc = calculate_forecast(data)
        current = fc.iloc[0]
        st.metric(
            f"{city} environmental suitability",
            f"{current['score']:.0f}/100",
            score_label(current["score"]),
        )

        fig = px.line(
            fc,
            x="date",
            y="score",
            markers=True,
            title="Seven-day environmental suitability trajectory",
        )
        fig.update_yaxes(range=[0, 100], title="Suitability score")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            fc[
                [
                    "date", "score", "rain_mm", "rain_7d_mm",
                    "temperature", "rain_probability"
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    except Exception as e:
        st.error(f"Weather service unavailable: {e}")

# ============================================================
# LIVE MAP
# ============================================================

elif tab_choice == "Live Map":
    st.title("🗺️ Environmental Suitability Map")
    st.caption(
        "Broad grid-cell research model. Individual occurrence coordinates "
        "are intentionally not displayed."
    )

    day = st.slider(
        "Forecast day",
        min_value=0,
        max_value=6,
        value=0,
        format="+%d days",
    )

    if st.button("Build / refresh research heatmap", type="primary"):
        map_df = build_research_map(day)
        if map_df.empty:
            st.error("No forecast cells could be retrieved.")
        else:
            st.session_state["map_df"] = map_df
            st.session_state["map_day"] = day

    map_df = st.session_state.get("map_df")

    if map_df is not None:
        m = folium.Map(location=[54.5, -2.5], zoom_start=6, tiles="OpenStreetMap")
        heat = [
            [r.lat, r.lon, r.score / 100.0]
            for r in map_df.itertuples()
        ]
        HeatMap(
            heat,
            radius=28,
            blur=24,
            min_opacity=0.25,
            max_zoom=8,
        ).add_to(m)

        st_folium(m, use_container_width=True, height=650)

        st.caption(
            "Heatmap represents modelled environmental suitability, not "
            "confirmed species presence."
        )
    else:
        st.info("Press the button to generate the current research map.")

# ============================================================
# 7 DAY FORECAST
# ============================================================

elif tab_choice == "7-Day Forecast":
    st.title("🔮 Seven-Day Forecast")

    cities = {
        "Manchester": (53.48, -2.24),
        "Liverpool": (53.41, -2.99),
        "Leeds": (53.80, -1.55),
        "Sheffield": (53.38, -1.47),
        "Birmingham": (52.49, -1.89),
        "Bristol": (51.45, -2.59),
        "Cardiff": (51.48, -3.18),
        "Edinburgh": (55.95, -3.19),
        "Glasgow": (55.86, -4.25),
        "London": (51.51, -0.13),
    }

    selected = st.selectbox("Reference location", list(cities))
    lat, lon = cities[selected]

    try:
        data = get_weather(lat, lon)
        fc = calculate_forecast(data)

        st.dataframe(
            fc,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            fc,
            x="date",
            y="score",
            text="score",
            title=f"{selected}: seven-day suitability forecast",
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        best = fc.loc[fc["score"].idxmax()]
        st.success(
            f"Forecast peak in this model: {best['date']} "
            f"at {best['score']:.0f}/100. "
            "This is an environmental model output, not a presence claim."
        )
    except Exception as e:
        st.error(f"Unable to retrieve forecast: {e}")

# ============================================================
# LOCATION EXPLORER
# ============================================================

elif tab_choice == "Location Explorer":
    st.title("📍 Location Explorer")

    st.write(
        "Enter coordinates for an area you want to analyse. "
        "The tool returns an environmental profile and forecast."
    )

    c1, c2 = st.columns(2)
    with c1:
        lat = st.number_input("Latitude", min_value=49.8, max_value=59.0, value=53.48)
    with c2:
        lon = st.number_input("Longitude", min_value=-7.8, max_value=1.8, value=-2.24)

    if st.button("Analyse location", type="primary"):
        try:
            data = get_weather(lat, lon)
            fc = calculate_forecast(data)

            current = fc.iloc[0]
            st.metric("Environmental suitability", f"{current['score']:.0f}/100")
            st.write(score_label(current["score"]))

            fig = px.line(fc, x="date", y="score", markers=True)
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Environmental drivers")
            drivers = pd.DataFrame({
                "Driver": ["Season", "Habitat prior", "Recent/forecast rainfall", "Temperature"],
                "Score": [
                    current["season_score"],
                    current["habitat_score"],
                    rain_score(current["rain_7d_mm"]),
                    temperature_score(current["temperature"]),
                ],
            })
            st.bar_chart(drivers.set_index("Driver"))

            st.info(
                "Navigation links are intentionally not generated from model "
                "predictions. The research interface should not be used as a "
                "collection locator."
            )
        except Exception as e:
            st.error(f"Unable to analyse location: {e}")

# ============================================================
# RESEARCH
# ============================================================

elif tab_choice == "Research":
    st.title("📊 Research & Validation")

    st.subheader("Planned research pipeline")

    pipeline = pd.DataFrame({
        "Stage": [
            "Occurrence data",
            "Weather history",
            "Habitat",
            "Terrain",
            "Feature engineering",
            "Baseline model",
            "Historical back-test",
            "Calibrated statistical model",
            "Machine-learning model",
        ],
        "Status": [
            "Next integration",
            "Available",
            "Available",
            "Next integration",
            "Next",
            "Current",
            "Next",
            "Future",
            "Future",
        ],
    })
    st.dataframe(pipeline, use_container_width=True, hide_index=True)

    st.info(
        "Occurrence data must be filtered by licence and spatial uncertainty "
        "before it is incorporated into the training set."
    )

    st.subheader("What we will measure")
    st.markdown(
        """
        - ROC-AUC
        - Precision / recall
        - Calibration
        - False positives / false negatives
        - Performance by region
        - Performance by month
        - Forecast accuracy at 1–7 day lead times
        - Contribution of weather, habitat and terrain variables
        """
    )

# ============================================================
# OTHER FUNGI
# ============================================================

elif tab_choice == "Other Fungi":
    st.title("🍄 Other Fungi")
    st.write(
        "The application is designed so additional species can use the same "
        "forecasting infrastructure."
    )

    species = st.selectbox(
        "Species",
        [
            "Chanterelle",
            "Morel",
            "Boletus species",
            "Fly agaric",
            "Custom research species",
        ],
    )

    st.info(
        f"{species}: species-specific habitat and phenology parameters will "
        "be added once appropriately licensed occurrence data are connected."
    )

# ============================================================
# MODEL
# ============================================================

elif tab_choice == "Model":
    st.title("⚙️ Model")

    st.subheader("Current baseline")
    st.code(
        """
Suitability =
    22% rainfall / moisture
  + 18% temperature
  + 12% humidity
  + 20% seasonality
  + 28% habitat

This is deliberately a transparent starting point.

It is NOT presented as a scientifically validated probability.
The next development stage is to calibrate the model against historical
UK occurrence records and perform temporal/spatial back-testing.
        """,
        language="text",
    )

    st.subheader("Planned predictors")
    predictors = [
        "1-day rainfall",
        "3-day rainfall",
        "7-day rainfall",
        "14-day rainfall",
        "rainfall persistence",
        "temperature mean/min/max",
        "relative humidity",
        "soil moisture",
        "soil temperature",
        "vapour-pressure deficit",
        "seasonality",
        "land-cover class",
        "land-cover classification probability",
        "elevation",
        "slope",
        "aspect",
        "historical occurrence density",
        "forecast uncertainty",
    ]
    st.write(predictors)

    st.warning(
        "Model outputs are environmental research estimates. They do not "
        "confirm species presence or provide identification advice."
    )

# ============================================================
# GUIDE
# ============================================================

elif tab_choice == "Guide":
    st.title("📖 User Guide")

    st.markdown(
        """
### What is this?

A research application for modelling environmental conditions associated
with *Psilocybe semilanceata* in the UK.

### What does the score mean?

The 0–100 score represents the model's estimate of **environmental
suitability**, not a percentage chance that mushrooms are present.

### Why use a forecast?

Fruiting response may lag changes in rainfall and other environmental
conditions. The application therefore evaluates a sequence of conditions
rather than only today's weather.

### Why is the map grid-based?

Species occurrence datasets can contain spatially generalised records.
A grid model is more honest than displaying an apparently precise mushroom
pin.

### What will improve the model?

The baseline model will eventually be calibrated using appropriately
licensed historical occurrence records and tested using historical
weather/forecast data.

### Important limitation

A favourable environmental score does not establish that a species occurs
at a location. Species identification also cannot safely be inferred from
the environmental model.
        """
    )

st.sidebar.divider()
st.sidebar.caption("Research build • model v0.1")
