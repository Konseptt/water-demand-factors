"""Run forecasting + feature-group ablation."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .ingest import load_data
from .clean import clean_data
from .features import add_features
from .models import build_models, chronological_split, score
from .ablate import feature_columns, ablation_labels

def run(input_path="data/sample/daily_demand_weather.csv", output_dir="results"):
    output = Path(output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = add_features(clean_data(load_data(input_path)))
    train, test = chronological_split(data)
    rows = []
    prediction_frame = pd.DataFrame({"date": test["date"], "actual": test["demand_proxy"]})
    for ablation in ablation_labels():
        drop_group = None if ablation == "none" else ablation.removeprefix("drop_")
        columns = feature_columns(drop_group)
        # Seasonal naive uses the prior week's observed demand and is always retained.
        baseline_pred = test["demand_lag_7"].to_numpy()
        baseline_metrics = score(test["demand_proxy"], baseline_pred)
        rows.append({"model": "SeasonalNaive", "ablation": ablation, **baseline_metrics})
        if ablation == "none":
            prediction_frame["SeasonalNaive"] = baseline_pred
        for name, model in build_models().items():
            model.fit(train[columns], train["demand_proxy"])
            pred = model.predict(test[columns])
            metrics = score(test["demand_proxy"], pred)
            rows.append({"model": name, "ablation": ablation, **metrics})
            if ablation == "none":
                prediction_frame[name] = pred
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False, float_format="%.6f")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 4.8))
    for column in ["actual", "SeasonalNaive", "Ridge", "RandomForest"]:
        plt.plot(prediction_frame["date"], prediction_frame[column], label=column, linewidth=1.7 if column == "actual" else 1.1)
    plt.title("One-day holdout: observed vs predicted demand proxy")
    plt.xlabel("Date"); plt.ylabel("Demand proxy")
    plt.legend(ncol=4, frameon=True); plt.tight_layout()
    plt.savefig(figures / "holdout_predictions.png", dpi=160); plt.close()
    ab = metrics[(metrics["model"] != "SeasonalNaive") & (metrics["ablation"] != "none")].copy()
    full = metrics[(metrics["model"] != "SeasonalNaive") & (metrics["ablation"] == "none")][["model", "mae"]].rename(columns={"mae": "full_mae"})
    ab = ab.merge(full, on="model")
    ab["mae_change_vs_full"] = ab["mae"] - ab["full_mae"]
    plt.figure(figsize=(9, 5))
    sns.barplot(data=ab, x="mae_change_vs_full", y="ablation", hue="model", orient="h")
    plt.axvline(0, color="black", linewidth=0.8); plt.xlabel("MAE change after dropping group (positive = worse)")
    plt.ylabel(""); plt.title("Feature-group ablation on the chronological holdout")
    plt.tight_layout(); plt.savefig(figures / "feature_ablation.png", dpi=160); plt.close()
    print(metrics[metrics["ablation"] == "none"].sort_values("mae").to_string(index=False))
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample/daily_demand_weather.csv")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run(args.input, args.output)

if __name__ == "__main__":
    main()
