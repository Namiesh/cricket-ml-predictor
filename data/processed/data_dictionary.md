# Data Dictionary: Player-Match Dataset (`player_match_data.csv`)

This document defines every feature/target column present in `data/processed/player_match_data.csv`.
It explicitly demarks **Exact Metrics** vs. **Approximate Metrics** resulting from structural limitations in the source dataset (`cricsheet_balls-arena`).

---

## 1. Identifiers & Participation Flags

| Column Name | Data Type | Classification | Description |
|---|---|---|---|
| `match_id` | `int64` | **Exact** | Unique identifier for each cricket match. |
| `player` | `string` | **Exact** | Cleaned name string of the player. |
| `matches_played` | `int64` | **Exact** | Indicator equal to `1` for each player-match observation. |
| `batting_matches` | `int64` | **Exact** | `1` if the player faced $\ge 1$ delivery as a striker in the match, else `0`. |
| `bowling_matches` | `int64` | **Exact** | `1` if the player bowled $\ge 1$ delivery in the match, else `0`. |

---

## 2. Batting Features & Targets

| Column Name | Data Type | Classification | Description & Limitations |
|---|---|---|---|
| `batting_runs` | `int64` | **Exact** | Sum of `runs_off_bat` scored by the striker. |
| `batting_boundaries` | `int64` | **Exact** | Total boundary hits (`batting_fours` + `batting_sixes`). |
| `batting_fours` | `int64` | **Exact** | Count of deliveries resulting in 4 runs off the bat. |
| `batting_sixes` | `int64` | **Exact** | Count of deliveries resulting in 6 runs off the bat. |
| `batting_dismissed` | `int64` | **Exact** | Total times the player was dismissed in the match (`player_dismissed == player`). |
| `batting_innings` | `int64` | **Exact** | Number of distinct innings in which the player batted during the match. |
| `batting_balls_approx` | `int64` | **Approximate** | Total deliveries faced as a striker. **Limitation:** Wide balls (which do not count as official balls faced in cricket) cannot be excluded because delivery extra types are aggregated into a single `extras` scalar. |
| `batting_strike_rate_approx` | `float64` | **Approximate** | $(	ext{batting\_runs} / 	ext{batting\_balls\_approx}) 	imes 100$. Inherits approximation from `batting_balls_approx`. |
| `batting_average_approx` | `float64` | **Approximate** | $	ext{batting\_runs} / 	ext{batting\_dismissed}$ (if dismissed $>0$, else `batting_runs`). Minor noise due to unrecorded retired hurt instances. |
| `balls_per_boundary_approx` | `float64` | **Approximate** | $	ext{batting\_balls\_approx} / 	ext{batting\_boundaries}$. `NaN` if no boundaries hit. Inherits approximation from `batting_balls_approx`. |

---

## 3. Bowling Features & Targets

| Column Name | Data Type | Classification | Description & Limitations |
|---|---|---|---|
| `bowling_runs_conceded_approx` | `int64` | **Approximate** | Sum of `runs_off_bat` + `extras` conceded while bowling. **Limitation:** In official cricket, fielding extras (byes and leg-byes) do not count against the bowler. Since extra types are unsegmented, all extras are attributed to the bowler. |
| `bowling_deliveries` | `int64` | **Exact** | Total deliveries bowled by the player in the match. |
| `bowling_wickets_approx` | `int64` | **Approximate** | Count of dismissals occurring during the bowler's deliveries. **Limitation:** In official cricket, run-outs do not count as bowler wickets. Because dismissal types are absent, all dismissals on the bowler's deliveries are counted. |
| `bowling_economy_rate_approx` | `float64` | **Approximate** | $(	ext{bowling\_runs\_conceded\_approx} / (	ext{bowling\_deliveries} / 6.0))$. Inherits approximation from unsegmented extras. |
| `bowling_strike_rate_approx` | `float64` | **Approximate** | $	ext{bowling\_deliveries} / 	ext{bowling\_wickets\_approx}$. `NaN` if zero wickets. Inherits approximation from `bowling_wickets_approx`. |
