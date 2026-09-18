"""Basic cleaning before features."""
import pandas as pd

def clean_data(frame):
    """Sort by date, drop duplicate dates, fill short gaps from the past only."""
    data = frame.copy()
    data = data.sort_values("date").drop_duplicates("date", keep="last")
    data = data.set_index("date")
    # Forward-fill only so values near the holdout cut never use future neighbors.
    data["demand_proxy"] = data["demand_proxy"].ffill(limit=2)
    for column in ["temp_c", "precip_mm"]:
        data[column] = data[column].ffill(limit=2)
    data["precip_mm"] = data["precip_mm"].clip(lower=0)
    data = data.dropna(subset=["demand_proxy", "temp_c", "precip_mm"])
    return data.reset_index()
