"""Daily feature engineering (lags/rolls use past demand only)."""
import pandas as pd
import numpy as np

FEATURE_GROUPS = {
    "lags": ["demand_lag_1", "demand_lag_7", "demand_lag_14"],
    "rolling": ["demand_roll_3", "demand_roll_7", "demand_roll_14"],
    "temp": ["temp_c", "temp_squared"],
    "precip": ["precip_mm", "precip_flag"],
    "dow": ["dow_sin", "dow_cos"],
    "month": ["month_sin", "month_cos"],
    "holiday": ["is_holiday"],
}

def _us_holidays(year):
    """US holiday dates for the demo (no extra package)."""
    jan1 = pd.Timestamp(year, 1, 1)
    jul4 = pd.Timestamp(year, 7, 4)
    dec25 = pd.Timestamp(year, 12, 25)
    # Memorial = last Monday in May; Labor = first Monday in Sep;
    # Thanksgiving = fourth Thursday in Nov.
    memorial = pd.Timestamp(year, 5, 31) - pd.Timedelta(days=(pd.Timestamp(year, 5, 31).weekday() - 0) % 7)
    labor = pd.Timestamp(year, 9, 1) + pd.Timedelta(days=(0 - pd.Timestamp(year, 9, 1).weekday()) % 7)
    thanksgiving = pd.Timestamp(year, 11, 1) + pd.Timedelta(days=(3 - pd.Timestamp(year, 11, 1).weekday()) % 7 + 21)
    holidays = {jan1, jul4, dec25, memorial, labor, thanksgiving}
    # US observed: Saturday holiday -> Friday; Sunday holiday -> Monday.
    holidays |= {day - pd.Timedelta(days=1) for day in list(holidays) if day.weekday() == 5}
    holidays |= {day + pd.Timedelta(days=1) for day in list(holidays) if day.weekday() == 6}
    return holidays

def add_features(frame):
    """Add predictors from weather/calendar and past demand only."""
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").drop_duplicates("date", keep="last")
    observed_dates = data["date"]
    data = data.set_index("date")
    # Calendar-day index so lags mean "N days ago", even if the Ohio join has gaps.
    daily = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq="D"))
    daily["demand_lag_1"] = daily["demand_proxy"].shift(1)
    daily["demand_lag_7"] = daily["demand_proxy"].shift(7)
    daily["demand_lag_14"] = daily["demand_proxy"].shift(14)
    prior = daily["demand_proxy"].shift(1)
    for window in (3, 7, 14):
        daily[f"demand_roll_{window}"] = prior.rolling(window, min_periods=window).mean()
    daily = daily.loc[observed_dates]
    daily["temp_squared"] = daily["temp_c"] ** 2
    daily["precip_flag"] = (daily["precip_mm"] > 0.1).astype(float)
    dow = daily.index.dayofweek
    daily["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    month = daily.index.month
    daily["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    daily["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    holiday_dates = set()
    for year in daily.index.year.unique():
        holiday_dates |= _us_holidays(int(year))
    daily["is_holiday"] = daily.index.isin(holiday_dates).astype(float)
    daily = daily.dropna(subset=sum(FEATURE_GROUPS.values(), []))
    return daily.reset_index(names="date")
