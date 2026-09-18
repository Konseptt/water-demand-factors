"""Models and chronological evaluation helpers."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def build_models():
    return {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=2.0)),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1)),
    }

def chronological_split(data, fraction=0.80):
    if len(data) < 2:
        raise ValueError("Need at least 2 rows for a chronological train/test split")
    cut = int(len(data) * fraction)
    cut = min(max(1, cut), len(data) - 1)
    return data.iloc[:cut].copy(), data.iloc[cut:].copy()

def score(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }
