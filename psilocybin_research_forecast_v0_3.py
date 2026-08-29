
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
        "Regional Outlook",
        "Location Explorer",
        "Research",
        "Validation",
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
    st.title("🍄 Research Forecast")
    st.caption(
        "UK environmental suitability dashboard for *Psilocybe semilanceata*."
    )

    city = st.selectbox(
        "Reference location",
        ["Manchester", "Liverpool", "Leeds", "Sheffield", "Birmingham",
         "Bristol", "Cardiff", "Edinburgh", "Glasgow", "London"],
        key="dashboard_city",
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

    try:
        lat, lon = coords[city]
        data = get_weather(lat, lon)
        fc = calculate_forecast(data)
        current = fc.iloc[0]
        trend, delta = trend_label(fc["score"])

        # Hero summary
        hero_l, hero_r = st.columns([1.7, 1])
        with hero_l:
            st.markdown(
                f"""
                <div style="
                    padding:22px;
                    border-radius:16px;
                    border:1px solid rgba(127,127,127,.18);
                    background:linear-gradient(135deg,rgba(127,127,127,.10),rgba(127,127,127,.04));
                ">
                    <div style="font-size:13px;opacity:.7;">TODAY • {city.upper()}</div>
                    <div style="font-size:48px;font-weight:800;line-height:1.05;margin-top:6px;">
                        {current['score']:.0f}<span style="font-size:20px;opacity:.55;"> / 100</span>
                    </div>
                    <div style="font-size:18px;font-weight:700;margin-top:5px;">
                        {score_band_short(current['score'])} environmental suitability
                    </div>
                    <div style="margin-top:10px;opacity:.75;">
                        Forecast trend: <b>{trend}</b>
                        {'(' + ('+' if delta >= 0 else '') + f'{delta:.0f} points over 7 days)' if delta else ''}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with hero_r:
            st.markdown("### Model status")
            st.success("🟢 Experimental baseline operational")
            st.caption("This is an environmental estimate, not a presence probability.")

        st.write("")

        # Key environmental drivers
        st.subheader("Today's environmental drivers")
        d1, d2, d3, d4 = st.columns(4)

        rain = rain_score(current["rain_7d_mm"])
        temp = temperature_score(current["temperature"])
        season = float(current["season_score"])
        habitat = float(current["habitat_score"])

        d1.metric("🌧️ Rainfall", f"{rain:.0f}/100")
        d2.metric("🌡️ Temperature", f"{temp:.0f}/100")
        d3.metric("🍂 Season", f"{season:.0f}/100")
        d4.metric("🌱 Habitat", f"{habitat:.0f}/100")

        st.subheader("Seven-day outlook")
        fig = px.area(
            fc,
            x="date",
            y="score",
            markers=True,
            title=None,
        )
        fig.update_yaxes(range=[0, 100], title="Environmental suitability")
        fig.update_xaxes(title="")
        fig.update_traces(
            hovertemplate="<b>%{x|%a %d %b}</b><br>Score: %{y:.0f}/100<extra></extra>"
        )
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast cards
        st.subheader("Day-by-day forecast")
        cards = st.columns(min(7, len(fc)))
        for col, (_, row) in zip(cards, fc.iterrows()):
            with col:
                st.markdown(
                    f"""
                    <div style="
                        padding:10px 7px;
                        text-align:center;
                        border-radius:12px;
                        border:1px solid rgba(127,127,127,.15);
                        background:rgba(127,127,127,.06);
                    ">
                        <div style="font-size:11px;opacity:.7;">{row['date'].strftime('%a')}</div>
                        <div style="font-size:12px;opacity:.7;">{row['date'].strftime('%d %b')}</div>
                        <div style="font-size:24px;font-weight:800;margin:4px 0;">{row['score']:.0f}</div>
                        <div style="font-size:11px;">{score_band_short(row['score'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.expander("🧠 Why is today's score what it is?"):
            drivers = pd.DataFrame({
                "Driver": ["Rainfall / moisture", "Temperature", "Seasonality", "Habitat"],
                "Score": [rain, temp, season, habitat],
            }).sort_values("Score", ascending=False)

            st.dataframe(
                drivers,
                use_container_width=True,
                hide_index=True,
            )

            strongest = drivers.iloc[0]
            weakest = drivers.iloc[-1]
            st.write(
                f"**Strongest current driver:** {strongest['Driver']} "
                f"({strongest['Score']:.0f}/100).  "
                f"**Most limiting driver:** {weakest['Driver']} "
                f"({weakest['Score']:.0f}/100)."
            )

        with st.expander("⚙️ Model status & limitations"):
            status = model_status()
            for key, value in status.items():
                st.write(f"**{key}:** {value}")
            st.info(
                "The current model is an experimental ecological baseline. "
                "It has not yet been calibrated against historical occurrence "
                "records and should not be interpreted as a validated probability."
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

    # --- Map controls -------------------------------------------------
    controls = st.container()
    with controls:
        c1, c2, c3 = st.columns([1.1, 1.1, 1.6])

        with c1:
            day = st.slider(
                "Forecast day",
                min_value=0,
                max_value=6,
                value=0,
                format="+%d days",
            )

        with c2:
            map_style = st.selectbox(
                "Map style",
                ["Clear grid", "Heatmap", "Satellite-style"],
                index=0,
            )

        with c3:
            st.markdown(
                """
                <div style="
                    padding: 10px 14px;
                    border-radius: 10px;
                    background: rgba(127,127,127,0.10);
                    margin-top: 4px;
                ">
                <b>What you're seeing</b><br>
                Environmental suitability from 0–100, not confirmed species presence.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Keep the requested day tied to the generated dataset.
    if st.button("🔄 Build / refresh research map", type="primary", use_container_width=True):
        with st.spinner("Building the environmental forecast map…"):
            map_df = build_research_map(day)

        if map_df.empty:
            st.error("No forecast cells could be retrieved.")
        else:
            st.session_state["map_df"] = map_df
            st.session_state["map_day"] = day

    map_df = st.session_state.get("map_df")

    if map_df is not None:
        stored_day = st.session_state.get("map_day", day)

        # --------------------------------------------------------------
        # Summary cards
        # --------------------------------------------------------------
        best_score = float(map_df["score"].max())
        mean_score = float(map_df["score"].mean())
        cells = len(map_df)

        a, b, c = st.columns(3)
        with a:
            st.metric("Highest model score", f"{best_score:.0f}/100")
        with b:
            st.metric("Average cell score", f"{mean_score:.0f}/100")
        with c:
            st.metric("Forecast day", f"+{stored_day} days")

        st.markdown(
            """
            <div style="
                display:flex;
                gap:8px;
                flex-wrap:wrap;
                margin: 8px 0 14px 0;
                font-size:0.86rem;
            ">
              <span style="padding:5px 10px;border-radius:999px;background:#e8f5e9;">
                🟢 80–100 Very high
              </span>
              <span style="padding:5px 10px;border-radius:999px;background:#f1f8e9;">
                🟡 65–79 High
              </span>
              <span style="padding:5px 10px;border-radius:999px;background:#fff8e1;">
                🟠 45–64 Moderate
              </span>
              <span style="padding:5px 10px;border-radius:999px;background:#fff3e0;">
                🔴 25–44 Low
              </span>
              <span style="padding:5px 10px;border-radius:999px;background:#ffebee;">
                ⚪ 0–24 Very low
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------------
        # Map
        # --------------------------------------------------------------
        tile_map = {
            "Clear grid": "CartoDB positron",
            "Heatmap": "OpenStreetMap",
            "Satellite-style": "CartoDB dark_matter",
        }

        m = folium.Map(
            location=[54.5, -2.5],
            zoom_start=6,
            tiles=tile_map[map_style],
            control_scale=True,
            prefer_canvas=True,
        )

        # Discrete research colour bands are intentionally used instead
        # of a blurred heatmap: this makes broad modelled cells much easier
        # to interpret without suggesting false precision.
        def score_colour(score):
            if score >= 80:
                return "#1b8a3a"
            if score >= 65:
                return "#70ad1f"
            if score >= 45:
                return "#e0a800"
            if score >= 25:
                return "#e8751a"
            return "#b7b7b7"

        def score_band(score):
            if score >= 80:
                return "Very high"
            if score >= 65:
                return "High"
            if score >= 45:
                return "Moderate"
            if score >= 25:
                return "Low"
            return "Very low"

        # Use broad rectangular cells. They communicate the model's spatial
        # resolution more honestly than a blurred heatmap.
        half = GRID_STEP / 2

        for r in map_df.itertuples():
            score = float(r.score)
            colour = score_colour(score)
            band = score_band(score)

            popup_html = f"""
            <div style="font-family:Arial,sans-serif;min-width:210px;">
                <div style="font-size:16px;font-weight:700;margin-bottom:6px;">
                    Environmental research cell
                </div>
                <div style="font-size:28px;font-weight:800;color:{colour};">
                    {score:.0f}<span style="font-size:14px;color:#666;">/100</span>
                </div>
                <div style="margin:4px 0 10px;"><b>{band}</b> environmental suitability</div>
                <hr style="border:0;border-top:1px solid #ddd;">
                <div>Rain over 7 days: <b>{r.rain_7d:.1f} mm</b></div>
                <div>Mean temperature: <b>{r.temp:.1f}°C</b></div>
                <div style="margin-top:8px;font-size:11px;color:#666;">
                    Broad modelled cell — not a confirmed occurrence location.
                </div>
            </div>
            """

            folium.Rectangle(
                bounds=[
                    [r.lat - half, r.lon - half],
                    [r.lat + half, r.lon + half],
                ],
                color=colour,
                weight=1,
                opacity=0.75,
                fill=True,
                fill_color=colour,
                fill_opacity=0.62,
                tooltip=f"{band} • {score:.0f}/100",
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(m)

        # Optional blurred heat layer for users who prefer the traditional
        # presentation. It is kept as a separate visual mode rather than
        # being the only representation.
        if map_style == "Heatmap":
            heat = [
                [r.lat, r.lon, r.score / 100.0]
                for r in map_df.itertuples()
            ]
            HeatMap(
                heat,
                radius=35,
                blur=22,
                min_opacity=0.30,
                max_zoom=8,
            ).add_to(m)

        # Add a compact HTML legend.
        legend = """
        <div style="
            position: fixed;
            bottom: 28px;
            left: 28px;
            z-index: 9999;
            background: rgba(255,255,255,0.96);
            padding: 12px 14px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,.18);
            font-family: Arial, sans-serif;
            font-size: 12px;
            line-height: 1.6;
        ">
          <div style="font-weight:700;margin-bottom:5px;">Environmental suitability</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#1b8a3a;margin-right:6px;"></span>80–100 Very high</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#70ad1f;margin-right:6px;"></span>65–79 High</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#e0a800;margin-right:6px;"></span>45–64 Moderate</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#e8751a;margin-right:6px;"></span>25–44 Low</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#b7b7b7;margin-right:6px;"></span>0–24 Very low</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend))

        st_folium(
            m,
            use_container_width=True,
            height=720,
            returned_objects=[],
        )

        st.caption(
            f"Showing {cells} broad research cells for forecast day +{stored_day}. "
            "Cell colours show the modelled environmental score; they do not "
            "represent confirmed species presence."
        )

        # --------------------------------------------------------------
        # Ranked research cells — useful for understanding the map without
        # pretending that a single point is an exact location.
        # --------------------------------------------------------------
        with st.expander("📊 View modelled cell summary"):
            summary = map_df.copy()
            summary["Band"] = summary["score"].apply(score_band)
            summary = (
                summary.sort_values("score", ascending=False)
                .reset_index(drop=True)
            )
            summary["Rank"] = np.arange(1, len(summary) + 1)

            st.dataframe(
                summary[
                    ["Rank", "Band", "score", "rain_7d", "temp"]
                ].rename(
                    columns={
                        "score": "Score",
                        "rain_7d": "7-day rain (mm)",
                        "temp": "Mean temp (°C)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.info(
            "Research safeguard: this map shows broad environmental suitability "
            "only. It is not a mushroom presence map, identification tool, or "
            "collection locator."
        )

    else:
        st.info(
            "Choose a forecast day and press **Build / refresh research map** "
            "to generate the environmental map."
        )

# ============================================================
# 7 DAY FORECAST
# ============================================================

elif tab_choice == "7-Day Forecast":
    st.title("🔮 Seven-Day Forecast")
    st.caption("How the environmental model changes over the coming week.")

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
        trend, delta = trend_label(fc["score"])
        best = fc.loc[fc["score"].idxmax()]

        a, b, c = st.columns(3)
        a.metric("Today", f"{fc.iloc[0]['score']:.0f}/100")
        b.metric("7-day peak", f"{best['score']:.0f}/100")
        c.metric("Overall trend", trend, f"{delta:+.0f} points")

        st.subheader(f"{selected}: environmental outlook")

        for _, row in fc.iterrows():
            score = float(row["score"])
            label = score_band_short(score)

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([1.4, 1.1, 1.2, 2.5])
                c1.markdown(f"**{row['date'].strftime('%A')}**  \n{row['date'].strftime('%d %B')}")
                c2.metric("Score", f"{score:.0f}/100")
                c3.write(f"**{label}**")
                c4.progress(int(score), text=f"{label} environmental suitability")

        fig = px.line(
            fc,
            x="date",
            y="score",
            markers=True,
            title="Seven-day suitability trajectory",
        )
        fig.update_yaxes(range=[0, 100], title="Score")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View weather inputs"):
            st.dataframe(
                fc[
                    ["date", "score", "rain_mm", "rain_7d_mm",
                     "temperature", "rain_probability"]
                ].rename(columns={
                    "score": "Suitability",
                    "rain_mm": "Rain (mm)",
                    "rain_7d_mm": "7-day rain (mm)",
                    "temperature": "Mean temp (°C)",
                    "rain_probability": "Rain probability (%)",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.success(
            f"Modelled peak: {best['date'].strftime('%A %d %B')} at "
            f"{best['score']:.0f}/100. This is an environmental model output, "
            "not a presence claim."
        )

    except Exception as e:
        st.error(f"Unable to retrieve forecast: {e}")

# ============================================================
# REGIONAL OUTLOOK
# ============================================================

elif tab_choice == "Regional Outlook":
    st.title("🌍 Regional Outlook")
    st.caption(
        "Broad regional comparison of the environmental baseline. "
        "Regions are summaries, not precise occurrence maps."
    )

    regions = {
        "Scotland": (56.0, -4.2),
        "Northern England": (54.7, -2.5),
        "Midlands": (52.7, -1.9),
        "Wales": (52.3, -3.7),
        "South West": (50.9, -3.5),
        "South East": (51.1, 0.2),
        "East of England": (52.4, 0.4),
    }

    rows = []
    for region, (lat, lon) in regions.items():
        try:
            data = get_weather(lat, lon)
            fc = calculate_forecast(data)
            trend, delta = trend_label(fc["score"])
            rows.append({
                "Region": region,
                "Today": float(fc.iloc[0]["score"]),
                "+3 days": float(fc.iloc[min(3, len(fc)-1)]["score"]),
                "+6 days": float(fc.iloc[-1]["score"]),
                "Trend": trend,
                "Change": delta,
            })
        except Exception:
            continue

    if rows:
        regional_df = pd.DataFrame(rows).sort_values("Today", ascending=False)

        st.subheader("Regional ranking")
        st.dataframe(
            regional_df.style.format({
                "Today": "{:.0f}",
                "+3 days": "{:.0f}",
                "+6 days": "{:.0f}",
                "Change": "{:+.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        chart_df = regional_df.melt(
            id_vars="Region",
            value_vars=["Today", "+3 days", "+6 days"],
            var_name="Forecast",
            value_name="Score",
        )
        fig = px.bar(
            chart_df,
            x="Region",
            y="Score",
            color="Forecast",
            barmode="group",
            title="Regional environmental suitability",
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Regional values are representative model calculations based on "
            "reference coordinates. They should not be interpreted as a "
            "complete assessment of every location within a region."
        )
    else:
        st.error("No regional forecast data could be retrieved.")


# ============================================================
# VALIDATION
# ============================================================

elif tab_choice == "Validation":
    st.title("🧪 Model Validation")
    st.caption(
        "A transparent framework for measuring whether the research model "
        "improves when historical occurrence data are connected."
    )

    st.subheader("Current validation status")

    v1, v2, v3 = st.columns(3)
    v1.metric("Model version", "v0.2")
    v2.metric("Historical records", "Not connected")
    v3.metric("Validated probability", "No")

    st.warning(
        "The current score is an ecological suitability index, not a calibrated "
        "probability. Validation metrics cannot honestly be reported until "
        "appropriate historical occurrence data are incorporated."
    )

    st.subheader("Planned validation metrics")
    validation = pd.DataFrame({
        "Metric": [
            "ROC-AUC",
            "Precision",
            "Recall",
            "Calibration",
            "False positives",
            "False negatives",
            "Regional performance",
            "Monthly performance",
            "1–7 day forecast accuracy",
        ],
        "Status": ["Pending"] * 9,
        "Purpose": [
            "Ranking ability",
            "Positive prediction quality",
            "Detection rate",
            "Probability reliability",
            "Overprediction analysis",
            "Underprediction analysis",
            "Spatial robustness",
            "Seasonal robustness",
            "Forecast lead-time performance",
        ],
    })
    st.dataframe(validation, use_container_width=True, hide_index=True)

    st.subheader("Validation pipeline")
    pipeline = [
        ("1", "Acquire appropriately licensed occurrence records"),
        ("2", "Apply spatial uncertainty and quality filters"),
        ("3", "Join historical weather and habitat variables"),
        ("4", "Create temporal/spatial training and test splits"),
        ("5", "Back-test the current baseline"),
        ("6", "Calibrate statistical model"),
        ("7", "Compare machine-learning alternatives"),
        ("8", "Report performance and uncertainty"),
    ]

    for number, step in pipeline:
        st.markdown(
            f"""
            <div style="
                display:flex;
                gap:12px;
                padding:9px 12px;
                margin:5px 0;
                border:1px solid rgba(127,127,127,.14);
                border-radius:10px;
            ">
                <b>{number}</b><span>{step}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# LOCATION EXPLORER
# ============================================================

elif tab_choice == "Location Explorer":
    st.title("📍 Location Explorer")
    st.caption(
        "Analyse broad environmental conditions for a coordinate. "
        "This is a research profile, not a collection locator."
    )

    c1, c2 = st.columns(2)
    with c1:
        lat = st.number_input(
            "Latitude",
            min_value=49.8,
            max_value=59.0,
            value=53.48,
            format="%.4f",
        )
    with c2:
        lon = st.number_input(
            "Longitude",
            min_value=-7.8,
            max_value=1.8,
            value=-2.24,
            format="%.4f",
        )

    if st.button("🔎 Analyse location", type="primary", use_container_width=True):
        try:
            with st.spinner("Analysing environmental conditions…"):
                data = get_weather(lat, lon)
                fc = calculate_forecast(data)

            current = fc.iloc[0]
            trend, delta = trend_label(fc["score"])

            a, b, c, d = st.columns(4)
            a.metric("Suitability", f"{current['score']:.0f}/100")
            b.metric("Trend", trend, f"{delta:+.0f}")
            c.metric("Rain, 7 days", f"{current['rain_7d_mm']:.1f} mm")
            d.metric("Mean temperature", f"{current['temperature']:.1f}°C")

            inject_score_badge(current["score"])

            st.subheader("Environmental profile")

            drivers = pd.DataFrame({
                "Driver": [
                    "Season",
                    "Habitat prior",
                    "Rainfall / moisture",
                    "Temperature",
                    "Humidity",
                ],
                "Score": [
                    current["season_score"],
                    current["habitat_score"],
                    rain_score(current["rain_7d_mm"]),
                    temperature_score(current["temperature"]),
                    current["humidity_score"],
                ],
            }).sort_values("Score", ascending=False)

            fig = px.bar(
                drivers,
                x="Score",
                y="Driver",
                orientation="h",
                text="Score",
                title="Current model drivers",
            )
            fig.update_xaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Additional environmental observations")

            e1, e2, e3 = st.columns(3)
            soil_m = current["soil_moisture"]
            soil_t = current["soil_temperature"]

            e1.metric(
                "Soil moisture",
                "—" if pd.isna(soil_m) else f"{soil_m:.3f}"
            )
            e2.metric(
                "Soil temperature",
                "—" if pd.isna(soil_t) else f"{soil_t:.1f}°C"
            )
            e3.metric(
                "Humidity",
                f"{current['humidity']:.0f}%"
            )

            st.subheader("Seven-day trajectory")
            fig = px.line(fc, x="date", y="score", markers=True)
            fig.update_yaxes(range=[0, 100], title="Suitability score")
            fig.update_xaxes(title="")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View raw environmental inputs"):
                st.dataframe(
                    fc[
                        [
                            "date", "score", "rain_3d_mm", "rain_7d_mm",
                            "temperature_min", "temperature",
                            "temperature_max", "humidity",
                            "soil_moisture", "soil_temperature",
                            "rain_probability",
                        ]
                    ].rename(columns={
                        "score": "Suitability",
                        "rain_3d_mm": "3-day rain (mm)",
                        "rain_7d_mm": "7-day rain (mm)",
                        "temperature_min": "Min temp (°C)",
                        "temperature": "Mean temp (°C)",
                        "temperature_max": "Max temp (°C)",
                        "humidity": "Humidity (%)",
                        "soil_moisture": "Soil moisture",
                        "soil_temperature": "Soil temp (°C)",
                        "rain_probability": "Rain probability (%)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

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

Additional diagnostics now available:
  • 3-day / 7-day rainfall
  • soil moisture
  • soil temperature
  • temperature min / max
  • humidity-derived dryness proxy

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
        "soil moisture (diagnostic in v0.2)",
        "soil temperature (diagnostic in v0.2)",
        "humidity-derived dryness proxy (diagnostic)",
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
st.sidebar.caption("Research build • model v0.2")
