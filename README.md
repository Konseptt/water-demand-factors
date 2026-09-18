# Water demand factors

I wanted a clear answer to one question: for one-day-ahead demand forecasts, what actually helps more, recent demand history, weather, calendar stuff, or holidays?

This repo is a small Python project (pandas, NumPy, scikit-learn). I train on earlier dates and test on later ones (no random shuffle), then drop feature groups one at a time to see which drops hurt MAE the most.

**Data note:** `data/sample/daily_demand_weather.csv` is fake/demo data I generated (365 days). It is not real utility meters. If you want a public Ohio stand-in, run `python -m src.fetch_public` (NOAA weather + USGS Scioto river flow).

## What is in here

- Features: lags, rolling demand, temp, precip, day of week, month, holidays
- Models: seasonal naive baseline, Ridge, Random Forest
- Metrics and ablation results in `results/metrics.csv`
- Extra feature ranking steps in `src/feature_selection.py`
- Optional public Ohio download in `src/fetch_public.py`

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.run_pipeline
```

CSV needs these columns: `date`, `demand_proxy`, `temp_c`, `precip_mm`.

Different file:

```bash
python -m src.run_pipeline --input path/to/daily.csv --output results
```

### Optional: Ohio public data

```bash
python -m src.fetch_public
python -m src.run_pipeline --input data/raw/ohio_public_daily.csv
python -m src.feature_selection --input data/raw/ohio_public_daily.csv
```

USGS `demand_proxy` here is river discharge in cfs, not billed water use. Details: [`docs/data_sources.md`](docs/data_sources.md).

## Method notes

- Last 20% of dates are held out for testing
- Rolling features use only past demand (shifted), so today is not used to predict today
- Numbers on the synthetic file are for checking the pipeline works. Re-run on real or public data before trusting the rankings

## Layout

```text
data/sample/   demo CSV
data/raw/      downloaded public data (optional)
docs/          notes on sources, ablation, feature selection
src/           code
results/       metrics + figures
```

## Docs

- [`docs/data_sources.md`](docs/data_sources.md) - where the data comes from
- [`docs/feature_ablation.md`](docs/feature_ablation.md) - what happened when I dropped feature groups
- [`docs/feature_selection.md`](docs/feature_selection.md) - other ranking methods I ran
