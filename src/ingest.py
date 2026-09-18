"""Load daily demand + weather CSV."""
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["date", "demand_proxy", "temp_c", "precip_mm"]

def load_data(path):
    """Load CSV and check required columns."""
    path = Path(path)
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("Every date must parse as a calendar date")
    for column in REQUIRED_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[REQUIRED_COLUMNS].copy()
