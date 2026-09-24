# Cricket Player ML Prediction System

An end-to-end Machine Learning Prediction Engine designed to forecast individual cricket player match performances (batting runs, strike rate, wickets, economy rate, etc.) using leakage-free historical features derived from ball-by-ball match data.

---

## Architecture & Data Pipeline

### 1. Dataset & Preprocessing
- **Source Data:** 7.14M ball-by-ball deliveries (`data/raw/cricsheet_balls-arena`, ~366 MB).
- **Inspection & Cleanup:** Handled in chunks without loading full raw CSV into memory (`scripts/inspect_dataset.py`).

### 2. Player-Match Aggregation
- **Output:** `data/processed/player_match_data.csv` (343,001 player-match records, 16,399 matches, 9,699 players).
- **Aggregation Script:** `scripts/build_player_match_dataset.py`.
- **Metrics:** Aggregates runs scored, boundaries, balls faced, innings, dismissals, bowling deliveries, runs conceded, and wickets taken per player per match.

### 3. Feature Engineering & Zero-Data-Leakage Guarantee
- **Output:** `data/processed/training_data.csv` (40 columns).
- **Feature Builder Script:** `scripts/build_features.py`.
- **31 Historical Features:** Includes career cumulative statistics and rolling window form (last 3, 5, 10 matches) for both batting and bowling.
- **Leakage Prevention:** Every feature for a match $M$ is calculated strictly using performance from matches occurring *before* $M$ (`shift(1)` + cumulative/rolling sums). The current match's performance is never used in feature calculation.

### 4. Model Training & Evaluation
- **Trainer Script:** `scripts/train_models.py`.
- **Model Architecture:** `scikit-learn` `HistGradientBoostingRegressor` (fast, native NaN feature support, tree-based).
- **Train/Test Methodology:** Strictly chronological split by `match_id` (first 80% matches = train set, last 20% matches = test set). No random shuffling.
- **Trained Targets:**
  1. `target_runs` (MAE: 13.96 vs Baseline: 17.88)
  2. `target_batting_average` (MAE: 12.11 vs Baseline: 15.48)
  3. `target_strike_rate` (MAE: 44.70 vs Baseline: 55.15)
  4. `target_balls_per_boundary` (MAE: 4.49 vs Baseline: 5.46)
  5. `target_wickets` (MAE: 0.76 vs Baseline: 0.81)
  6. `target_economy_rate` (MAE: 2.09 vs Baseline: 3.72, $R^2 = 0.4805$)
  7. `target_bowling_strike_rate` (MAE: 8.64 vs Baseline: 10.63)

---

## How to Run Predictions

### 1. Run Unit & Integration Tests
```bash
python scripts/test_prediction_engine.py
```

### 2. Predict Performance for a Single Player
```bash
python scripts/predict_players.py --player "da warner"
```

### 3. Run Batch Predictions for All Known Players
```bash
python scripts/predict_players.py --all
```

---

## Programmatic Usage

```python
from src.prediction.predictor import CricketPredictor

predictor = CricketPredictor()

# Predict player's next match performance
pred = predictor.predict_player("v kohli")
print(pred)

# Leakage-free prediction for a specific historical match point
pred_hist = predictor.predict_player_for_match("v kohli", match_id=1082591)
```

---

## Model Limitations
- **Match-Level Variance:** Single-match outcomes in cricket have high intrinsic variance.
- **Unsegmented Extras/Dismissals:** Wides/no-balls and run-outs/bowler-wickets are aggregated in the raw source dataset, making strike rates and bowler wickets defensible approximations.
- **Non-Negativity Clipping:** Model predictions are raw regression values; post-processing applies physical non-negativity bounds ($\ge 0.0$).
