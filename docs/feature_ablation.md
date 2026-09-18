# Feature ablation

I fit each model with all feature groups, then drop one group at a time and re-score on the chronological holdout. Numbers are from `results/metrics.csv` on the synthetic demo. The plot is `results/figures/feature_ablation.png`.

## Holdout metrics

| Model | Ablation | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| SeasonalNaive | none | 4.443 | 5.272 | -0.372 |
| **Ridge** | **none (full)** | **2.252** | **2.770** | **0.621** |
| RandomForest | none | 3.650 | 4.437 | 0.028 |
| Ridge | drop_lags | 2.381 | 2.867 | 0.594 |
| Ridge | drop_rolling | 2.312 | 2.844 | 0.601 |
| Ridge | drop_temp | 2.963 | 3.575 | 0.369 |
| Ridge | drop_precip | 2.453 | 3.108 | 0.523 |
| Ridge | drop_dow | 2.383 | 3.057 | 0.539 |
| Ridge | drop_month | 3.431 | 4.286 | 0.093 |
| Ridge | drop_holiday | 2.422 | 2.992 | 0.558 |
| RandomForest | drop_lags | 3.309 | 3.981 | 0.218 |
| RandomForest | drop_rolling | 3.581 | 4.349 | 0.067 |
| RandomForest | drop_temp | 3.641 | 4.399 | 0.045 |
| RandomForest | drop_precip | 3.715 | 4.558 | -0.025 |
| RandomForest | drop_dow | 3.801 | 4.621 | -0.054 |
| RandomForest | drop_month | 3.779 | 4.615 | -0.051 |
| RandomForest | drop_holiday | 3.656 | 4.441 | 0.027 |

SeasonalNaive stays the same across ablations because it only uses demand from 7 days ago.

## Ridge (best full model)

If MAE goes up after I drop a group, that group was useful on this holdout.

| Dropped group | Ridge MAE | ΔMAE vs full (2.252) | Rank |
|---|---:|---:|---:|
| month | 3.431 | **+1.178** | 1 |
| temp | 2.963 | **+0.711** | 2 |
| precip | 2.453 | +0.201 | 3 |
| holiday | 2.422 | +0.169 | 4 |
| dow | 2.383 | +0.130 | 5 |
| lags | 2.381 | +0.129 | 6 |
| rolling | 2.312 | +0.060 | 7 |

On this demo, month and temperature hurt Ridge the most when removed. Lags and rolling still help a bit, but less once weather and calendar are in.

## Random Forest notes

Full RF is worse than Ridge here (MAE 3.650, R² 0.028). Dropping lags actually improves RF a little (MAE 3.309). With one short synthetic year, trees can latch onto noisy lag structure, so I would not treat RF ranks as the main answer.

## Takeaways

1. Compare against a time holdout and a seasonal-naive baseline
2. Report MAE change vs the full model, not only importance scores
3. These ranks are from synthetic data. Re-run on Ohio public series (or real utility data) before saying what moves demand in the real world
