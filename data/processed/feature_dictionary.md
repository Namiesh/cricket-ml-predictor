# Feature Dictionary: Training Dataset (`training_data.csv`)

This document defines all features and targets in `data/processed/training_data.csv`.
Every feature is explicitly classified as **Historical Feature**, **Current-Match Target**, or **Approximate Metric**.

---

## 1. Identifiers

| Column Name | Type | Classification | Description |
|---|---|---|---|
| `match_id` | `int64` | Identifier | Unique match ID. |
| `player` | `string` | Identifier | Cleaned player name string. |

---

## 2. Batting Historical Features (Strictly BEFORE Current Match)

| Column Name | Type | Classification | Description & Limitations |
|---|---|---|---|
| `previous_matches` | `int64` | Historical Feature | Total matches played prior to current match. |
| `previous_batting_matches` | `int64` | Historical Feature | Matches where player faced $\ge 1$ ball prior to current match. |
| `career_runs_before_match` | `int64` | Historical Feature | Cumulative batting runs scored prior to current match. |
| `career_dismissals_before_match` | `int64` | Historical Feature | Cumulative dismissals prior to current match. |
| `career_average_before_match` | `float64` | Historical Feature | Career batting average prior to current match. |
| `career_strike_rate_before_match` | `float64` | Historical / Approx | Career strike rate prior to current match. *(Approximate: wide balls unseparated)*. |
| `career_boundaries_before_match` | `int64` | Historical Feature | Cumulative boundary hits prior to current match. |
| `runs_last_3` | `int64` | Historical Feature | Total runs scored in last 3 previous batting matches. |
| `runs_last_5` | `int64` | Historical Feature | Total runs scored in last 5 previous batting matches. |
| `runs_last_10` | `int64` | Historical Feature | Total runs scored in last 10 previous batting matches. |
| `batting_average_last_3` | `float64` | Historical Feature | Batting average over last 3 previous batting matches. |
| `batting_average_last_5` | `float64` | Historical Feature | Batting average over last 5 previous batting matches. |
| `batting_average_last_10` | `float64` | Historical Feature | Batting average over last 10 previous batting matches. |
| `strike_rate_last_3` | `float64` | Historical / Approx | Strike rate over last 3 previous batting matches. |
| `strike_rate_last_5` | `float64` | Historical / Approx | Strike rate over last 5 previous batting matches. |
| `strike_rate_last_10` | `float64` | Historical / Approx | Strike rate over last 10 previous batting matches. |
| `boundaries_last_5` | `int64` | Historical Feature | Total boundaries hit in last 5 previous batting matches. |
| `balls_per_boundary_last_5` | `float64` | Historical / Approx | Balls per boundary in last 5 previous batting matches (`NaN` if no boundaries). |

---

## 3. Bowling Historical Features (Strictly BEFORE Current Match)

| Column Name | Type | Classification | Description & Limitations |
|---|---|---|---|
| `previous_bowling_matches` | `int64` | Historical Feature | Matches where player bowled $\ge 1$ delivery prior to current match. |
| `career_wickets_before_match` | `int64` | Historical / Approx | Cumulative wickets before current match. *(Approximate: includes run-outs)*. |
| `career_bowling_runs_before_match` | `int64` | Historical / Approx | Cumulative runs conceded before current match. *(Approximate: includes fielding extras)*. |
| `career_economy_before_match` | `float64` | Historical / Approx | Career economy rate before current match. |
| `career_bowling_strike_rate_before_match` | `float64` | Historical / Approx | Career bowling strike rate before current match. |
| `wickets_last_3` | `int64` | Historical / Approx | Wickets taken in last 3 previous bowling matches. |
| `wickets_last_5` | `int64` | Historical / Approx | Wickets taken in last 5 previous bowling matches. |
| `wickets_last_10` | `int64` | Historical / Approx | Wickets taken in last 10 previous bowling matches. |
| `economy_last_3` | `float64` | Historical / Approx | Economy rate over last 3 previous bowling matches. |
| `economy_last_5` | `float64` | Historical / Approx | Economy rate over last 5 previous bowling matches. |
| `economy_last_10` | `float64` | Historical / Approx | Economy rate over last 10 previous bowling matches. |
| `bowling_strike_rate_last_5` | `float64` | Historical / Approx | Bowling strike rate over last 5 previous bowling matches. |
| `bowling_strike_rate_last_10` | `float64` | Historical / Approx | Bowling strike rate over last 10 previous bowling matches. |

---

## 4. Current-Match Targets (Current Match Performance)

| Column Name | Type | Classification | Description |
|---|---|---|---|
| `target_runs` | `int64` | Current-Match Target | Runs scored in current match. |
| `target_batting_average` | `float64` | Current-Match Target | Batting average in current match. |
| `target_strike_rate` | `float64` | Current-Match Target / Approx | Strike rate in current match. |
| `target_balls_per_boundary` | `float64` | Current-Match Target / Approx | Balls per boundary in current match. |
| `target_wickets` | `int64` | Current-Match Target / Approx | Wickets taken in current match. |
| `target_economy_rate` | `float64` | Current-Match Target / Approx | Economy rate in current match. |
| `target_bowling_strike_rate` | `float64` | Current-Match Target / Approx | Bowling strike rate in current match. |
