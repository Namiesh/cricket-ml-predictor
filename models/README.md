# Trained Machine Learning Models (`models/`)

This directory contains trained regression models and associated registry metadata for the **Cricket Player ML Prediction System**.

---

## Model Artifacts

| Model File | Prediction Target | Model Architecture | Training Samples | Test Samples | Test MAE | Test $R^2$ |
|---|---|---|---|---|---|---|
| `runs_model.pkl` | `target_runs` | `HistGradientBoostingRegressor` | 274,891 | 68,110 | 13.9564 | 0.2434 |
| `batting_average_model.pkl` | `target_batting_average` | `HistGradientBoostingRegressor` | 274,891 | 68,110 | 12.1122 | 0.2074 |
| `strike_rate_model.pkl` | `target_strike_rate` | `HistGradientBoostingRegressor` | 274,891 | 68,110 | 44.7019 | 0.2332 |
| `balls_per_boundary_model.pkl` | `target_balls_per_boundary` | `HistGradientBoostingRegressor` | 138,187 | 31,365 | 4.4929 | 0.1262 |
| `wickets_model.pkl` | `target_wickets` | `HistGradientBoostingRegressor` | 274,891 | 68,110 | 0.7608 | 0.2451 |
| `economy_rate_model.pkl` | `target_economy_rate` | `HistGradientBoostingRegressor` | 274,891 | 68,110 | 2.0926 | 0.4805 |
| `bowling_strike_rate_model.pkl` | `target_bowling_strike_rate` | `HistGradientBoostingRegressor` | 97,994 | 24,335 | 8.6358 | 0.0702 |

---

## Feature Schema

Every model accepts the exact **31 numerical feature columns** documented in `model_metadata.json` and `model_registry.json`.
Features consist strictly of pre-match historical career totals and rolling window statistics (last 3, 5, 10 matches). No target columns, match IDs, or current-match data are included.

---

## Loading Models Programmatically

```python
import pickle
import json

# Load metadata
with open("models/model_registry.json", "r") as f:
    registry = json.load(f)

# Load runs model
with open("models/runs_model.pkl", "rb") as f:
    runs_model = pickle.load(f)

# Predict given a 1-row DataFrame X containing the 31 feature columns
predicted_runs = runs_model.predict(X)
```

---

## Prediction Limitations
1. **Match-Level Variance:** Single-innings cricket performance has high natural variance. Predictions represent expected value given historical form.
2. **Unsegmented Extras & Dismissals:** Raw data aggregates extras and dismissals, making strike rates and bowler wickets defensible approximations.
3. **Non-Negativity Clipping:** Model predictions are raw regression values; post-processing applies physical non-negativity bounds ($\ge 0.0$).
