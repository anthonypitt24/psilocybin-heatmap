# Psilocybin Research Forecast

A Streamlit research application for modelling broad environmental suitability for *Psilocybe semilanceata* in the UK.

## Current version

v0.1 — transparent ecological baseline.

### Current data connection
- Open-Meteo forecast API

### Planned research integrations
- NBN Atlas occurrence data, filtered by licence and spatial uncertainty
- UKCEH Land Cover Map 2024
- terrain/elevation data
- historical weather and historical forecast runs
- model calibration and back-testing

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

No API key is required for the initial Open-Meteo connection.

## Important

The map is intentionally grid-based and reports environmental suitability rather than individual occurrence locations.
