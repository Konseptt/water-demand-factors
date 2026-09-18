"""Feature ranking: correlation, MI, RFE, permutation importance.

Group ablation lives in src/ablate.py / run_pipeline. This writes
results/feature_selection.csv.
"""
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .ingest import load_data
from .clean import clean_data
from .features import FEATURE_GROUPS, add_features
from .models import chronological_split
from .ablate import feature_columns


def _all_features():
    return feature_columns(None)


def run(input_path: str = "data/sample/daily_demand_weather.csv", output_dir: str = "results") -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = add_features(clean_data(load_data(input_path)))
    train, _test = chronological_split(data)
    cols = _all_features()
    rows = []

    # 1) Correlation (train only)
    for col in cols:
        rows.append({
            "method": "correlation",
            "feature": col,
            "score": float(train[col].corr(train["demand_proxy"])),
            "rank_note": "pearson_vs_target_train",
        })
        rows.append({
            "method": "spearman",
            "feature": col,
            "score": float(train[col].corr(train["demand_proxy"], method="spearman")),
            "rank_note": "spearman_vs_target_train",
        })

    # 2) Mutual information
    mi = mutual_info_regression(
        train[cols].to_numpy(),
        train["demand_proxy"].to_numpy(),
        random_state=0,
    )
    for col, score in zip(cols, mi):
        rows.append({
            "method": "mutual_info",
            "feature": col,
            "score": float(score),
            "rank_note": "mi_regression_train",
        })

    # 4) RFE with Ridge (after scaling)
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=0)),
    ])
    # sklearn RFE needs an estimator with coef_; use the Ridge step via a thin wrapper
    from sklearn.linear_model import Ridge as RidgeEst
    est = RidgeEst(alpha=1.0, random_state=0)
    X_train = StandardScaler().fit_transform(train[cols])
    rfe = RFE(estimator=est, n_features_to_select=max(5, len(cols) // 2), step=1)
    rfe.fit(X_train, train["demand_proxy"])
    for col, rank, selected in zip(cols, rfe.ranking_, rfe.support_):
        rows.append({
            "method": "rfe",
            "feature": col,
            "score": float(-rank),  # higher (less negative) = selected earlier
            "rank_note": "selected" if selected else f"rank_{int(rank)}",
        })

    # 5) Permutation importance on a train-internal holdout (not the final test).
    # Keep this diagnostic-only; do not pick features from these ranks then rescore test.
    train_fit, train_val = chronological_split(train)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=0)),
    ])
    model.fit(train_fit[cols], train_fit["demand_proxy"])
    perm = permutation_importance(
        model,
        train_val[cols],
        train_val["demand_proxy"],
        n_repeats=20,
        random_state=0,
        scoring="neg_mean_absolute_error",
    )
    for col, score, std in zip(cols, perm.importances_mean, perm.importances_std):
        rows.append({
            "method": "permutation",
            "feature": col,
            "score": float(score),
            "rank_note": f"train_val_std_{std:.6f}",
        })

    feature_to_group = {
        feat: group for group, feats in FEATURE_GROUPS.items() for feat in feats
    }
    out = pd.DataFrame(rows)
    out["feature_group"] = out["feature"].map(feature_to_group)
    out.to_csv(output / "feature_selection.csv", index=False, float_format="%.6f")
    print(out.groupby("method")["feature"].count().to_string())
    print(f"Wrote {output / 'feature_selection.csv'}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample/daily_demand_weather.csv")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
