# Feature selection

I wanted more than one way to see which features matter. Everything below is pandas / NumPy / scikit-learn.

## Steps I ran

| Step | Method | What it answers | Where it lands |
|---|---|---|---|
| 1 | Pearson / Spearman vs `demand_proxy` (train only) | Do they move together? | `results/feature_selection.csv` |
| 2 | Mutual information | Any dependence, including nonlinear | same CSV |
| 3 | Group ablation (`src/ablate.py`) | If I drop this group, how much worse is holdout MAE? | `results/metrics.csv` |
| 4 | RFE with Ridge | Smaller feature set under a linear model | same CSV |
| 5 | Permutation importance on a train-internal holdout | Which features the fitted model leans on (diagnostic; not the final test) | same CSV |

```bash
python -m src.feature_selection
python -m src.feature_selection --input data/raw/ohio_public_daily.csv
```

## How I read conflicts

- Ablation tells you if a whole group is needed. Permutation is a train-side diagnostic of model sensitivity. For “what drives performance,” I lean on ablation.
- High mutual info with low correlation can mean a nonlinear weather or calendar effect still worth keeping.
- Ranks on `data/sample/` are for the demo file. After `fetch_public`, re-run and update claims from the new CSV.

## Demo snapshot

On the synthetic run, dropping month then temp hurt Ridge most. Lags and rolling mattered less once calendar and weather were in. Longer write-up: `docs/feature_ablation.md`.
