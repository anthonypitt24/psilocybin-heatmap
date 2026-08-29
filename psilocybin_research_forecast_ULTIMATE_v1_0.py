
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
# UI THEME
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }

        [data-testid="stMetric"] {
            background: rgba(127,127,127,0.08);
            border: 1px solid rgba(127,127,127,0.16);
            padding: 12px 14px;
            border-radius: 12px;
        }

        div[data-testid="stButton"] > button {
            border-radius: 10px;
            font-weight: 650;
            min-height: 44px;
        }

        div[data-testid="stSelectbox"] > div,
        div[data-testid="stNumberInput"] > div {
            border-radius: 10px;
        }

        .stAlert {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
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
    """Cached forecast request for a single coordinate."""
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


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_weather_batch(points, forecast_days=7):
    """Fetch multiple coordinates in one Open-Meteo request where supported."""
    if not points:
        return []

    lats = ",".join(str(round(float(p[0]), 2)) for p in points)
    lons = ",".join(str(round(float(p[1]), 2)) for p in points)

    params = {
        "latitude": lats,
        "longitude": lons,
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

    r = requests.get(OPEN_METEO_FORECAST, params=params, timeout=45)
    r.raise_for_status()
    payload = r.json()

    # Open-Meteo returns a list for multi-coordinate requests and a dict
    # for a single coordinate.
    return payload if isinstance(payload, list) else [payload]


def daily_weather_frame(data):
    d = pd.DataFrame(data["daily"])
    d["date"] = pd.to_datetime(d["time"]).dt.date

    # Add daily summaries from hourly data where available.
    hourly = data.get("hourly", {})
    if hourly and "time" in hourly:
        h = pd.DataFrame(hourly)
        h["time"] = pd.to_datetime(h["time"])
        h["date"] = h["time"].dt.date

        for source, target in [
            ("relative_humidity_2m", "humidity_mean"),
            ("soil_moisture_0_to_7cm", "soil_moisture_mean"),
            ("soil_temperature_0_to_7cm", "soil_temperature_mean"),
            ("wind_speed_10m", "wind_mean"),
        ]:
            if source in h.columns:
                daily_mean = h.groupby("date")[source].mean()
                d[target] = d["date"].map(daily_mean)

    return d


def vpd_proxy_score(humidity):
    """Simple humidity-based atmospheric dryness proxy for the baseline."""
    if pd.isna(humidity):
        return 65.0
    # Higher humidity generally means lower atmospheric dryness.
    return clamp(100.0 - max(0.0, float(humidity) - 60.0) * 1.8, 20, 100)


def soil_moisture_score(value):
    if pd.isna(value):
        return 60.0
    # Open-Meteo soil moisture is volumetric water content.
    # This is deliberately a broad baseline, not a species-calibrated optimum.
    return clamp(normal_score(float(value), 0.05, 0.18, 0.38, 0.55))


def soil_temperature_score(value):
    if pd.isna(value):
        return 60.0
    return clamp(normal_score(float(value), 1, 7, 14, 22))


def calculate_forecast(data, habitat="Grassland"):
    daily = daily_weather_frame(data)

    daily["rain_3d"] = daily["rain_sum"].rolling(3, min_periods=1).sum()
    daily["rain_7d"] = daily["rain_sum"].rolling(7, min_periods=1).sum()

    rows = []
    for _, r in daily.iterrows():
        month = r["date"].month

        humidity = float(r.get("humidity_mean", 75.0) or 75.0)
        soil_moisture = r.get("soil_moisture_mean", np.nan)
        soil_temperature = r.get("soil_temperature_mean", np.nan)

        parts = {
            "rain": rain_score(r["rain_7d"]),
            "temperature": temperature_score(r["temperature_2m_mean"]),
            "humidity": clamp(normal_score(humidity, 45, 70, 95, 100)),
            "season": season_score(month),
            "habitat": habitat_score(habitat),
        }

        # These are currently diagnostic variables. We expose them in the
        # research interface while retaining the transparent v0.2 weighting.
        score = weighted_score(parts)

        rows.append({
            "date": r["date"],
            "score": round(score, 1),
            "rain_3d_mm": round(float(r["rain_3d"]), 1),
            "rain_7d_mm": round(float(r["rain_7d"]), 1),
            "rain_mm": round(float(r["rain_sum"]), 1),
            "temperature": round(float(r["temperature_2m_mean"]), 1),
            "temperature_min": round(float(r["temperature_2m_min"]), 1),
            "temperature_max": round(float(r["temperature_2m_max"]), 1),
            "humidity": round(humidity, 1),
            "soil_moisture": (
                round(float(soil_moisture), 3)
                if not pd.isna(soil_moisture) else np.nan
            ),
            "soil_temperature": (
                round(float(soil_temperature), 1)
                if not pd.isna(soil_temperature) else np.nan
            ),
            "vpd_proxy_score": round(vpd_proxy_score(humidity), 1),
            "soil_moisture_score": round(soil_moisture_score(soil_moisture), 1),
            "soil_temperature_score": round(
                soil_temperature_score(soil_temperature), 1
            ),
            "rain_probability": int(
                r.get("precipitation_probability_max", 0) or 0
            ),
            "season_score": parts["season"],
            "habitat_score": parts["habitat"],
            "humidity_score": parts["humidity"],
        })

    return pd.DataFrame(rows)


def build_research_map(forecast_date_offset=0):
    """Build a broad environmental map using batched, cached weather calls."""
    points = make_grid()
    target = date.today() + timedelta(days=forecast_date_offset)

    # Still deliberately sampled/coarse to keep the research map broad.
    sampled = points[::4]
    rows = []

    batch_size = 40
    batches = [
        sampled[i:i + batch_size]
        for i in range(0, len(sampled), batch_size)
    ]

    progress = st.progress(0, text="Building environmental forecast…")

    for batch_index, batch in enumerate(batches):
        try:
            payloads = get_weather_batch(batch, 7)

            for (lat, lon), weather in zip(batch, payloads):
                df = calculate_forecast(weather)
                match = df[df["date"] == target]

                if match.empty:
                    idx = min(forecast_date_offset, len(df) - 1)
                    match = df.iloc[[idx]]

                r = match.iloc[0]

                rows.append({
                    "lat": lat,
                    "lon": lon,
                    "score": float(r["score"]),
                    "rain_7d": float(r["rain_7d_mm"]),
                    "temp": float(r["temperature"]),
                    "soil_moisture": r["soil_moisture"],
                    "soil_temperature": r["soil_temperature"],
                })

        except Exception:
            # Continue building the map if one batch fails.
            continue

        progress.progress(
            (batch_index + 1) / max(1, len(batches)),
            text=f"Building environmental forecast… "
                 f"{batch_index + 1}/{len(batches)} batches"
        )

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

def score_band_short(score):
    if score >= 80:
        return "Very high"
    if score >= 65:
        return "High"
    if score >= 45:
        return "Moderate"
    if score >= 25:
        return "Low"
    return "Very low"


def trend_label(scores):
    """Describe the direction of the forecast without implying certainty."""
    if len(scores) < 2:
        return "Stable", 0.0
    delta = float(scores.iloc[-1] - scores.iloc[0])
    if delta >= 8:
        return "Improving", delta
    if delta <= -8:
        return "Declining", delta
    return "Stable", delta


def model_status():
    return {
        "Weather data": "Available",
        "Habitat prior": "Available",
        "Historical occurrence data": "Not yet connected",
        "Statistical calibration": "Not yet completed",
        "Current model": "Experimental baseline",
        "Model version": "v0.2",
    }


def inject_score_badge(score):
    band = score_band_short(score)
    st.markdown(
        f"""
        <div style="
            display:inline-flex;
            align-items:center;
            gap:10px;
            padding:7px 13px;
            border-radius:999px;
            background:rgba(127,127,127,.10);
            border:1px solid rgba(127,127,127,.16);
            font-weight:650;
            margin-bottom:10px;
        ">
            <span style="font-size:18px;">🍄</span>
            <span>{band} environmental suitability</span>
            <span style="opacity:.65;">{score:.0f}/100</span>
        </div>
        """,
        unsafe_allow_html=True,
    )





# ============================================================
# FINAL RELEASE UI — RESEARCH FORECAST v1.0
# ============================================================

APP_VERSION = "1.0"
MODEL_VERSION = "v0.2 baseline"

CITIES = {
    "Manchester": (53.48, -2.24), "Liverpool": (53.41, -2.99),
    "Leeds": (53.80, -1.55), "Sheffield": (53.38, -1.47),
    "Birmingham": (52.49, -1.89), "Bristol": (51.45, -2.59),
    "Cardiff": (51.48, -3.18), "Edinburgh": (55.95, -3.19),
    "Glasgow": (55.86, -4.25), "London": (51.51, -0.13),
}

REGIONS = {
    "Scotland": (56.0, -4.2), "Northern England": (54.7, -2.5),
    "Midlands": (52.7, -1.9), "Wales": (52.3, -3.7),
    "South West": (50.9, -3.5), "South East": (51.1, 0.2),
    "East of England": (52.4, 0.4),
}


def final_band(score):
    if score >= 80: return "Very high"
    if score >= 65: return "High"
    if score >= 45: return "Moderate"
    if score >= 25: return "Low"
    return "Very low"


def final_colour(score):
    if score >= 80: return "#1b8a3a"
    if score >= 65: return "#70ad1f"
    if score >= 45: return "#e0a800"
    if score >= 25: return "#e8751a"
    return "#8f8f8f"


def final_trend(series):
    if len(series) < 2: return "Stable", 0.0
    change = float(series.iloc[-1] - series.iloc[0])
    if change >= 8: return "Improving", change
    if change <= -8: return "Declining", change
    return "Stable", change


def driver_frame(row):
    return pd.DataFrame({
        "Driver": ["Rainfall / moisture", "Temperature", "Humidity",
                   "Seasonality", "Habitat prior"],
        "Score": [
            rain_score(row["rain_7d_mm"]),
            temperature_score(row["temperature"]),
            row.get("humidity_score", 75.0),
            row["season_score"],
            row["habitat_score"],
        ],
    }).sort_values("Score", ascending=False)


def forecast_figure(fc, title=None):
    fig = px.area(fc, x="date", y="score", markers=True, title=title)
    fig.update_yaxes(range=[0, 100], title="Environmental suitability")
    fig.update_xaxes(title="")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
    return fig


st.markdown(
    """
    <style>
      .block-container {max-width:1500px;padding-top:1.3rem;padding-bottom:2.5rem;}
      [data-testid="stMetric"] {
        border:1px solid rgba(127,127,127,.16);
        border-radius:12px;padding:10px 13px;background:rgba(127,127,127,.06);
      }
      div[data-testid="stButton"] > button {
        border-radius:10px;min-height:42px;font-weight:650;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🍄 Research Forecast")
st.sidebar.caption("UK environmental modelling platform")
st.sidebar.caption(f"Release {APP_VERSION} • {MODEL_VERSION}")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Forecast Map", "7-Day Forecast", "Regional Outlook",
     "Location Explorer", "Historical Research", "Validation",
     "Species", "Model", "Guide"],
)

st.sidebar.divider()
st.sidebar.info(
    "Broad environmental research model. Outputs do not verify species "
    "presence or provide a collection locator."
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    st.title("🍄 Research Forecast")
    st.caption("Today's environmental picture, seven-day trajectory and model state.")

    city = st.selectbox("Reference location", list(CITIES), key="dash_city")

    try:
        lat, lon = CITIES[city]
        with st.spinner("Loading forecast…"):
            fc = calculate_forecast(get_weather(lat, lon))

        current = fc.iloc[0]
        trend, change = final_trend(fc["score"])
        peak = fc.loc[fc["score"].idxmax()]

        left, right = st.columns([1.6, 1])
        with left:
            colour = final_colour(current["score"])
            st.markdown(
                f"""
                <div style="padding:22px;border-radius:16px;
                border:1px solid rgba(127,127,127,.18);
                background:rgba(127,127,127,.06);">
                <div style="font-size:12px;opacity:.65;">TODAY • {city.upper()}</div>
                <div style="font-size:50px;font-weight:800;">
                {current['score']:.0f}<span style="font-size:20px;opacity:.5;"> / 100</span>
                </div>
                <div style="font-size:18px;font-weight:700;color:{colour};">
                {final_band(current['score'])} environmental suitability
                </div></div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown("### Model status")
            st.success("🟢 Forecast engine operational")
            st.caption("Experimental baseline — not a calibrated probability.")

        a, b, c, d = st.columns(4)
        a.metric("7-day peak", f"{peak['score']:.0f}/100")
        b.metric("Trend", trend, f"{change:+.0f} points")
        c.metric("7-day rainfall", f"{current['rain_7d_mm']:.1f} mm")
        d.metric("Mean temperature", f"{current['temperature']:.1f}°C")

        st.subheader("Environmental drivers")
        drivers = driver_frame(current)
        cols = st.columns(len(drivers))
        for col, (_, row) in zip(cols, drivers.iterrows()):
            col.metric(row["Driver"], f"{row['Score']:.0f}/100")

        st.subheader("Seven-day trajectory")
        st.plotly_chart(forecast_figure(fc), use_container_width=True)

        st.subheader("Day-by-day outlook")
        cards = st.columns(7)
        for col, (_, row) in zip(cards, fc.iterrows()):
            with col:
                st.markdown(
                    f"""
                    <div style="text-align:center;padding:10px 4px;border-radius:12px;
                    border:1px solid rgba(127,127,127,.15);background:rgba(127,127,127,.05);">
                    <div style="font-size:11px;opacity:.7;">{row['date'].strftime('%a')}</div>
                    <div style="font-size:11px;opacity:.7;">{row['date'].strftime('%d %b')}</div>
                    <div style="font-size:25px;font-weight:800;">{row['score']:.0f}</div>
                    <div style="font-size:10px;">{final_band(row['score'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.expander("🧠 Why is today's score what it is?"):
            st.dataframe(drivers, use_container_width=True, hide_index=True)
            st.write(
                f"Strongest driver: **{drivers.iloc[0]['Driver']}**. "
                f"Most limiting driver: **{drivers.iloc[-1]['Driver']}**."
            )
    except Exception as e:
        st.error(f"Weather service unavailable: {e}")

# ============================================================
# FORECAST MAP
# ============================================================

elif page == "Forecast Map":
    st.title("🗺️ Forecast Map")
    st.caption("Broad grid-cell environmental suitability; no individual occurrence coordinates.")

    c1, c2 = st.columns(2)
    with c1:
        day = st.slider("Forecast day", 0, 6, 0, format="+%d days")
    with c2:
        style = st.selectbox("Visualisation", ["Clear grid", "Heatmap", "Dark grid"])

    if st.button("🔄 Build / refresh map", type="primary", use_container_width=True):
        with st.spinner("Building environmental forecast map…"):
            df = build_research_map(day)
        if df.empty:
            st.error("No forecast cells could be retrieved.")
        else:
            st.session_state["ultimate_map"] = df
            st.session_state["ultimate_map_day"] = day

    map_df = st.session_state.get("ultimate_map")
    if map_df is not None:
        stored_day = st.session_state.get("ultimate_map_day", day)
        a, b, c = st.columns(3)
        a.metric("Highest cell", f"{map_df['score'].max():.0f}/100")
        b.metric("Average cell", f"{map_df['score'].mean():.0f}/100")
        c.metric("Forecast", f"+{stored_day} days")

        tiles = {"Clear grid": "CartoDB positron",
                 "Heatmap": "OpenStreetMap",
                 "Dark grid": "CartoDB dark_matter"}
        m = folium.Map(location=[54.5, -2.5], zoom_start=6,
                       tiles=tiles[style], control_scale=True, prefer_canvas=True)

        half = GRID_STEP / 2
        for r in map_df.itertuples():
            score = float(r.score)
            colour = final_colour(score)
            popup = (
                f"<b>Environmental research cell</b><br>"
                f"<div style='font-size:26px;font-weight:800;color:{colour};'>"
                f"{score:.0f}/100</div><b>{final_band(score)}</b><br><br>"
                f"7-day rain: {r.rain_7d:.1f} mm<br>"
                f"Mean temperature: {r.temp:.1f}°C<hr>"
                f"<small>Broad modelled cell, not a confirmed occurrence.</small>"
            )
            folium.Rectangle(
                bounds=[[r.lat-half, r.lon-half], [r.lat+half, r.lon+half]],
                color=colour, weight=1, opacity=.75, fill=True,
                fill_color=colour, fill_opacity=.62,
                tooltip=f"{final_band(score)} • {score:.0f}/100",
                popup=folium.Popup(popup, max_width=300),
            ).add_to(m)

        if style == "Heatmap":
            HeatMap(
                [[r.lat, r.lon, r.score / 100] for r in map_df.itertuples()],
                radius=35, blur=22, min_opacity=.3, max_zoom=8
            ).add_to(m)

        legend = """
        <div style="position:fixed;bottom:25px;left:25px;z-index:9999;
        background:rgba(255,255,255,.96);padding:12px 14px;border-radius:10px;
        box-shadow:0 2px 10px rgba(0,0,0,.18);font:12px Arial;line-height:1.7;">
        <b>Environmental suitability</b><br>
        🟢 80–100 Very high<br>🟡 65–79 High<br>🟠 45–64 Moderate<br>
        🔴 25–44 Low<br>⚪ 0–24 Very low
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend))
        st_folium(m, use_container_width=True, height=720, returned_objects=[])

        with st.expander("📊 Ranked map cells"):
            ranked = map_df.sort_values("score", ascending=False).copy()
            ranked.insert(0, "Rank", range(1, len(ranked)+1))
            ranked["Band"] = ranked["score"].apply(final_band)
            st.dataframe(
                ranked[["Rank","Band","score","rain_7d","temp"]].rename(
                    columns={"score":"Score","rain_7d":"7-day rain (mm)",
                             "temp":"Mean temp (°C)"}
                ),
                use_container_width=True, hide_index=True
            )
    else:
        st.info("Choose a forecast day and build the map.")

# ============================================================
# 7-DAY FORECAST
# ============================================================

elif page == "7-Day Forecast":
    st.title("🔮 Seven-Day Forecast")
    selected = st.selectbox("Reference location", list(CITIES))
    try:
        lat, lon = CITIES[selected]
        fc = calculate_forecast(get_weather(lat, lon))
        trend, change = final_trend(fc["score"])
        peak = fc.loc[fc["score"].idxmax()]

        a, b, c = st.columns(3)
        a.metric("Today", f"{fc.iloc[0]['score']:.0f}/100")
        b.metric("Peak", f"{peak['score']:.0f}/100")
        c.metric("Trend", trend, f"{change:+.0f}")

        st.plotly_chart(forecast_figure(fc, f"{selected}: seven-day suitability"),
                        use_container_width=True)

        for _, row in fc.iterrows():
            with st.container(border=True):
                x1, x2, x3, x4 = st.columns([1.5,1,1.2,2.5])
                x1.markdown(f"**{row['date'].strftime('%A')}**  \n{row['date'].strftime('%d %B')}")
                x2.metric("Score", f"{row['score']:.0f}/100")
                x3.write(f"**{final_band(row['score'])}**")
                x4.progress(int(row["score"]), text="Suitability")

        with st.expander("View environmental inputs"):
            st.dataframe(fc, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Unable to retrieve forecast: {e}")

# ============================================================
# REGIONAL OUTLOOK
# ============================================================

elif page == "Regional Outlook":
    st.title("🌍 Regional Outlook")
    st.caption("Broad representative regional summaries.")

    rows = []
    progress = st.progress(0, text="Loading regional forecasts…")
    for i, (region, (lat, lon)) in enumerate(REGIONS.items()):
        try:
            fc = calculate_forecast(get_weather(lat, lon))
            trend, change = final_trend(fc["score"])
            rows.append({
                "Region": region,
                "Today": fc.iloc[0]["score"],
                "+3 days": fc.iloc[min(3, len(fc)-1)]["score"],
                "+6 days": fc.iloc[-1]["score"],
                "Trend": trend,
                "Change": change,
            })
        except Exception:
            pass
        progress.progress((i+1) / len(REGIONS))
    progress.empty()

    if rows:
        rdf = pd.DataFrame(rows).sort_values("Today", ascending=False)
        st.dataframe(
            rdf.style.format({"Today":"{:.0f}", "+3 days":"{:.0f}",
                              "+6 days":"{:.0f}", "Change":"{:+.0f}"}),
            use_container_width=True, hide_index=True
        )
        chart = rdf.melt(id_vars="Region",
                         value_vars=["Today","+3 days","+6 days"],
                         var_name="Forecast", value_name="Score")
        fig = px.bar(chart, x="Region", y="Score", color="Forecast",
                     barmode="group", title="Regional comparison")
        fig.update_yaxes(range=[0,100])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No regional forecast data could be retrieved.")

# ============================================================
# LOCATION EXPLORER
# ============================================================

elif page == "Location Explorer":
    st.title("📍 Location Explorer")
    st.caption("Environmental analysis for a coordinate.")

    a, b = st.columns(2)
    with a:
        lat = st.number_input("Latitude", 49.8, 59.0, 53.48, format="%.4f")
    with b:
        lon = st.number_input("Longitude", -7.8, 1.8, -2.24, format="%.4f")

    if st.button("🔎 Analyse location", type="primary", use_container_width=True):
        try:
            with st.spinner("Analysing…"):
                fc = calculate_forecast(get_weather(lat, lon))
            current = fc.iloc[0]
            trend, change = final_trend(fc["score"])

            a,b,c,d = st.columns(4)
            a.metric("Suitability", f"{current['score']:.0f}/100")
            b.metric("Trend", trend, f"{change:+.0f}")
            c.metric("7-day rain", f"{current['rain_7d_mm']:.1f} mm")
            d.metric("Mean temp", f"{current['temperature']:.1f}°C")

            drivers = driver_frame(current)
            fig = px.bar(drivers, x="Score", y="Driver", orientation="h",
                         text="Score", title="Environmental driver scores")
            fig.update_xaxes(range=[0,100])
            st.plotly_chart(fig, use_container_width=True)

            e1,e2,e3 = st.columns(3)
            e1.metric("Soil moisture", "—" if pd.isna(current["soil_moisture"])
                      else f"{current['soil_moisture']:.3f}")
            e2.metric("Soil temperature", "—" if pd.isna(current["soil_temperature"])
                      else f"{current['soil_temperature']:.1f}°C")
            e3.metric("Humidity", f"{current['humidity']:.0f}%")

            st.plotly_chart(forecast_figure(fc, "Seven-day location trajectory"),
                            use_container_width=True)
        except Exception as e:
            st.error(f"Unable to analyse location: {e}")

# ============================================================
# HISTORICAL RESEARCH
# ============================================================

elif page == "Historical Research":
    st.title("📚 Historical Research")
    st.caption("The research layer for historical weather and occurrence back-testing.")

    st.warning(
        "Historical occurrence records are not connected in this release. "
        "No historical presence claims or performance figures are fabricated."
    )

    st.subheader("Research pipeline")
    st.dataframe(
        pd.DataFrame({
            "Stage": [
                "Licensed occurrence records", "Spatial uncertainty filtering",
                "Historical weather", "Habitat / terrain", "Feature engineering",
                "Temporal / spatial splits", "Baseline back-test",
                "Calibrated model", "Independent validation"
            ],
            "Status": [
                "Pending", "Pending", "Available source", "Next integration",
                "Next", "Next", "Next", "Future", "Future"
            ],
        }),
        use_container_width=True, hide_index=True
    )

# ============================================================
# VALIDATION
# ============================================================

elif page == "Validation":
    st.title("🧪 Validation & Model Health")

    a,b,c = st.columns(3)
    a.metric("Model", MODEL_VERSION)
    b.metric("Validated probability", "No")
    c.metric("Occurrence dataset", "Not connected")

    st.warning(
        "The 0–100 value is an environmental suitability index, not a "
        "calibrated probability."
    )

    st.subheader("Validation metrics")
    st.dataframe(
        pd.DataFrame({
            "Metric": [
                "ROC-AUC","Precision","Recall","Calibration",
                "False positives","False negatives",
                "Regional performance","Monthly performance",
                "1–7 day forecast accuracy"
            ],
            "Status": ["Pending"] * 9,
        }),
        use_container_width=True, hide_index=True
    )

# ============================================================
# SPECIES
# ============================================================

elif page == "Species":
    st.title("🍄 Species Models")
    species = st.selectbox(
        "Species",
        ["Psilocybe semilanceata","Chanterelle","Morel",
         "Boletus species","Fly agaric"]
    )
    if species == "Psilocybe semilanceata":
        st.success("Primary research species • baseline model active")
        st.write(
            "The active baseline uses rainfall/moisture, temperature, humidity, "
            "seasonality and habitat."
        )
    else:
        st.info(
            f"{species}: species-specific parameters are not calibrated yet. "
            "This selection is an architectural placeholder."
        )

# ============================================================
# MODEL
# ============================================================

elif page == "Model":
    st.title("⚙️ Model & Methodology")
    st.subheader("Current baseline")
    st.code(
        """
Suitability =
    22% rainfall / moisture
  + 18% temperature
  + 12% humidity
  + 20% seasonality
  + 28% habitat

Additional diagnostics:
  • 3-day / 7-day rainfall
  • temperature min / mean / max
  • soil moisture
  • soil temperature
  • humidity-derived dryness proxy

The 0–100 score is an environmental suitability index,
not a scientifically validated probability.
        """,
        language="text"
    )

    st.subheader("Future calibration variables")
    st.write([
        "1 / 3 / 7 / 14-day rainfall", "Rainfall persistence",
        "Temperature range", "Relative humidity", "Soil moisture",
        "Soil temperature", "Vapour-pressure deficit", "Seasonality",
        "Land cover", "Elevation / slope / aspect",
        "Historical occurrence density", "Forecast uncertainty"
    ])

# ============================================================
# GUIDE
# ============================================================

elif page == "Guide":
    st.title("📖 User Guide")
    st.subheader("What does the score mean?")
    st.write(
        "The 0–100 score estimates broad environmental suitability. It is not "
        "the percentage chance that a species is present."
    )
    st.subheader("Why a grid map?")
    st.write(
        "Broad cells avoid false precision and are more appropriate for "
        "generalised research data."
    )
    st.subheader("How should I use the forecast?")
    st.write(
        "Look at the trend and environmental drivers as well as the headline score."
    )
    st.subheader("What is still missing?")
    st.write(
        "The major scientific step is historical occurrence data, followed by "
        "spatial/temporal back-testing and calibration."
    )
    st.warning(
        "Environmental suitability does not establish species occurrence "
        "or provide identification advice."
    )

st.sidebar.divider()
st.sidebar.caption("Research Forecast • Final UI release v1.0")
