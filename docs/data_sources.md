# Data sources

## Demo CSV (default)

`data/sample/daily_demand_weather.csv` is synthetic Columbus-like data (365 days). I keep it as the default so the pipeline runs offline with no API calls.

Columns: `date`, `demand_proxy`, `temp_c`, `precip_mm`.

## Public Ohio data

```bash
python -m src.fetch_public
python -m src.run_pipeline --input data/raw/ohio_public_daily.csv
```

| Role | Series | ID | Links |
|---|---|---|---|
| Weather | NCEI GHCND TMAX/TMIN/PRCP | `USW00014821` (John Glenn Columbus) | [station](https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USW00014821/detail), [bulk csv.gz](https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00014821.csv.gz) |
| Target proxy | USGS daily mean discharge (cfs, `00060`) | `03227500` Scioto at Columbus | [inventory](https://waterdata.usgs.gov/nwis/inventory/?site_no=03227500) |

Another nearby gage if needed: Olentangy at Columbus `03227107`.

### Units

- GHCND temps are tenths of C, so `temp_c = (TMAX + TMIN) / 20`
- GHCND precip is tenths of mm, so `precip_mm = PRCP / 10`
- USGS `00060` is cfs. I store it as `demand_proxy` and label it as discharge, not retail demand

### Join rules

1. Inner join weather and discharge on calendar date
2. Drop rows missing any required column
3. Write a short retrieval note to `data/raw/ohio_public_retrieval.txt`
4. Same-day weather is not a forecast. Real ops would need forecast vintage

### Important

USGS flow is river discharge near Columbus. It is not Columbus Water AMI/SCADA. I used public series because meter data is proprietary. The question I ask on top of that join is which feature groups move a short-horizon forecast of the proxy.

## What I did not use

- Utility AMI/SCADA (proprietary)
- Weather forecasts without issuance time (that would make skill look better than it is)
- Random municipal dumps with no retrieval date

For system context only (not the daily label): [EPA SDWIS](https://www.epa.gov/enviro/sdwis-search), [Columbus Division of Water](https://www.columbus.gov/Services/Columbus-Water-Power/About-Columbus-Water-Power/The-Division-of-Water).
