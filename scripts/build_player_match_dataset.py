import os
import math
import pandas as pd
import numpy as np

def find_dataset():
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    if not os.path.exists(raw_dir):
        raw_dir = os.path.join("data", "raw")
    
    files = [
        f for f in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, f)) and not f.startswith(".")
    ]
    csv_files = [f for f in files if f.endswith(".csv")]
    target_file = csv_files[0] if csv_files else files[0]
    return os.path.join(raw_dir, target_file)

def create_data_dictionary(output_path):
    content = """# Data Dictionary: Player-Match Dataset (`player_match_data.csv`)

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
| `batting_strike_rate_approx` | `float64` | **Approximate** | $(\text{batting\_runs} / \text{batting\_balls\_approx}) \times 100$. Inherits approximation from `batting_balls_approx`. |
| `batting_average_approx` | `float64` | **Approximate** | $\text{batting\_runs} / \text{batting\_dismissed}$ (if dismissed $>0$, else `batting_runs`). Minor noise due to unrecorded retired hurt instances. |
| `balls_per_boundary_approx` | `float64` | **Approximate** | $\text{batting\_balls\_approx} / \text{batting\_boundaries}$. `NaN` if no boundaries hit. Inherits approximation from `batting_balls_approx`. |

---

## 3. Bowling Features & Targets

| Column Name | Data Type | Classification | Description & Limitations |
|---|---|---|---|
| `bowling_runs_conceded_approx` | `int64` | **Approximate** | Sum of `runs_off_bat` + `extras` conceded while bowling. **Limitation:** In official cricket, fielding extras (byes and leg-byes) do not count against the bowler. Since extra types are unsegmented, all extras are attributed to the bowler. |
| `bowling_deliveries` | `int64` | **Exact** | Total deliveries bowled by the player in the match. |
| `bowling_wickets_approx` | `int64` | **Approximate** | Count of dismissals occurring during the bowler's deliveries. **Limitation:** In official cricket, run-outs do not count as bowler wickets. Because dismissal types are absent, all dismissals on the bowler's deliveries are counted. |
| `bowling_economy_rate_approx` | `float64` | **Approximate** | $(\text{bowling\_runs\_conceded\_approx} / (\text{bowling\_deliveries} / 6.0))$. Inherits approximation from unsegmented extras. |
| `bowling_strike_rate_approx` | `float64` | **Approximate** | $\text{bowling\_deliveries} / \text{bowling\_wickets\_approx}$. `NaN` if zero wickets. Inherits approximation from `bowling_wickets_approx`. |
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Created data dictionary at: {output_path}")

def build_player_match_dataset():
    input_path = find_dataset()
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "player_match_data.csv")
    dict_md = os.path.join(output_dir, "data_dictionary.md")

    print(f"Reading raw dataset from: {input_path}")
    print("Processing 7.14M rows in memory-efficient chunks (chunksize=250,000)...")

    chunksize = 250_000
    batting_list = []
    dismissal_list = []
    bowling_list = []
    innings_list = []

    usecols = ['match_id', 'innings', 'striker', 'bowler', 'runs_off_bat', 'extras', 'player_dismissed']

    for chunk in pd.read_csv(input_path, chunksize=chunksize, usecols=usecols, low_memory=False):
        # Normalize and clean player names
        chunk['striker'] = chunk['striker'].astype(str).str.strip()
        chunk['bowler'] = chunk['bowler'].astype(str).str.strip()

        valid_striker = chunk['striker'].notna() & (chunk['striker'] != '') & (chunk['striker'] != 'nan') & (chunk['striker'] != 'None')
        valid_bowler = chunk['bowler'].notna() & (chunk['bowler'] != '') & (chunk['bowler'] != 'nan') & (chunk['bowler'] != 'None')

        # 1. Batting Aggregations
        df_bat = chunk[valid_striker]
        if not df_bat.empty:
            bat_grp = df_bat.groupby(['match_id', 'striker']).agg(
                batting_runs=('runs_off_bat', 'sum'),
                batting_fours=('runs_off_bat', lambda x: (x == 4).sum()),
                batting_sixes=('runs_off_bat', lambda x: (x == 6).sum()),
                batting_balls_approx=('runs_off_bat', 'count')
            ).reset_index().rename(columns={'striker': 'player'})
            batting_list.append(bat_grp)

            # Innings tracking
            inn_grp = df_bat[['match_id', 'striker', 'innings']].drop_duplicates().rename(columns={'striker': 'player'})
            innings_list.append(inn_grp)

        # 2. Dismissal Aggregations
        df_dism = chunk[chunk['player_dismissed'].notna()].copy()
        if not df_dism.empty:
            df_dism['player_dismissed'] = df_dism['player_dismissed'].astype(str).str.strip()
            valid_dism = (df_dism['player_dismissed'] != '') & (df_dism['player_dismissed'] != 'nan') & (df_dism['player_dismissed'] != 'None')
            df_dism = df_dism[valid_dism]
            if not df_dism.empty:
                dism_grp = df_dism.groupby(['match_id', 'player_dismissed']).size().reset_index(name='batting_dismissed').rename(columns={'player_dismissed': 'player'})
                dismissal_list.append(dism_grp)

        # 3. Bowling Aggregations
        df_bowl = chunk[valid_bowler]
        if not df_bowl.empty:
            df_bowl_calc = df_bowl.copy()
            df_bowl_calc['total_runs'] = df_bowl_calc['runs_off_bat'] + df_bowl_calc['extras']
            df_bowl_calc['is_wicket'] = df_bowl_calc['player_dismissed'].notna().astype(int)

            bowl_grp = df_bowl_calc.groupby(['match_id', 'bowler']).agg(
                bowling_runs_conceded_approx=('total_runs', 'sum'),
                bowling_deliveries=('total_runs', 'count'),
                bowling_wickets_approx=('is_wicket', 'sum')
            ).reset_index().rename(columns={'bowler': 'player'})
            bowling_list.append(bowl_grp)

    print("Merging chunk statistics into unified player-match records...")
    df_batting_all = pd.concat(batting_list, ignore_index=True).groupby(['match_id', 'player'], as_index=False).sum()
    df_innings_all = pd.concat(innings_list, ignore_index=True).drop_duplicates().groupby(['match_id', 'player'], as_index=False).agg(batting_innings=('innings', 'count'))
    df_dismissals_all = pd.concat(dismissal_list, ignore_index=True).groupby(['match_id', 'player'], as_index=False).sum()
    df_bowling_all = pd.concat(bowling_list, ignore_index=True).groupby(['match_id', 'player'], as_index=False).sum()

    # Outer join across all roles to prevent loss of player records
    df_merged = pd.merge(df_batting_all, df_innings_all, on=['match_id', 'player'], how='outer')
    df_merged = pd.merge(df_merged, df_dismissals_all, on=['match_id', 'player'], how='outer')
    df_all = pd.merge(df_merged, df_bowling_all, on=['match_id', 'player'], how='outer')

    # Fill default zeroes for missing numerical counts
    fill_cols = {
        'batting_runs': 0,
        'batting_fours': 0,
        'batting_sixes': 0,
        'batting_balls_approx': 0,
        'batting_innings': 0,
        'batting_dismissed': 0,
        'bowling_runs_conceded_approx': 0,
        'bowling_deliveries': 0,
        'bowling_wickets_approx': 0
    }
    df_all = df_all.fillna(fill_cols)

    for col in fill_cols.keys():
        df_all[col] = df_all[col].astype(int)

    # Calculate Derived Batter Metrics
    df_all['batting_boundaries'] = df_all['batting_fours'] + df_all['batting_sixes']

    df_all['batting_strike_rate_approx'] = np.where(
        df_all['batting_balls_approx'] > 0,
        (df_all['batting_runs'] / df_all['batting_balls_approx']) * 100.0,
        0.0
    ).round(2)

    df_all['batting_average_approx'] = np.where(
        df_all['batting_dismissed'] > 0,
        df_all['batting_runs'] / df_all['batting_dismissed'],
        df_all['batting_runs'].astype(float)
    ).round(2)

    df_all['balls_per_boundary_approx'] = np.where(
        df_all['batting_boundaries'] > 0,
        (df_all['batting_balls_approx'] / df_all['batting_boundaries']).round(2),
        np.nan
    )

    # Calculate Derived Bowling Metrics
    df_all['bowling_economy_rate_approx'] = np.where(
        df_all['bowling_deliveries'] > 0,
        (df_all['bowling_runs_conceded_approx'] / (df_all['bowling_deliveries'] / 6.0)).round(2),
        0.0
    )

    df_all['bowling_strike_rate_approx'] = np.where(
        df_all['bowling_wickets_approx'] > 0,
        (df_all['bowling_deliveries'] / df_all['bowling_wickets_approx']).round(2),
        np.nan
    )

    # Participation Flags
    df_all['matches_played'] = 1
    df_all['batting_matches'] = (df_all['batting_balls_approx'] > 0).astype(int)
    df_all['bowling_matches'] = (df_all['bowling_deliveries'] > 0).astype(int)

    # Sort strictly by match_id and player name
    df_all = df_all.sort_values(by=['match_id', 'player']).reset_index(drop=True)

    # Reorder columns logically
    cols_order = [
        'match_id',
        'player',
        'matches_played',
        'batting_matches',
        'bowling_matches',
        'batting_runs',
        'batting_boundaries',
        'batting_fours',
        'batting_sixes',
        'batting_dismissed',
        'batting_innings',
        'batting_balls_approx',
        'batting_strike_rate_approx',
        'batting_average_approx',
        'balls_per_boundary_approx',
        'bowling_runs_conceded_approx',
        'bowling_deliveries',
        'bowling_wickets_approx',
        'bowling_economy_rate_approx',
        'bowling_strike_rate_approx'
    ]
    df_all = df_all[cols_order]

    # Write output CSV
    print(f"Writing dataset to: {output_csv}")
    df_all.to_csv(output_csv, index=False)

    # Write Data Dictionary MD
    create_data_dictionary(dict_md)

    print("\n--- VALIDATION REPORT ---")
    print(f"1. Total Row Count: {len(df_all):,}")
    print(f"2. Number of Columns: {len(df_all.columns)}")
    print("3. Columns List:")
    for idx, c in enumerate(df_all.columns, 1):
        print(f"   {idx}. {c}")

    print("\n4. Basic Validation Checks:")
    print(f"   - Duplicate (match_id, player) pairs: {df_all.duplicated(subset=['match_id', 'player']).sum()}")
    print(f"   - Null or empty player names: {df_all['player'].isna().sum() + (df_all['player'] == '').sum()}")
    print(f"   - Total distinct matches: {df_all['match_id'].nunique():,}")
    print(f"   - Total distinct players: {df_all['player'].nunique():,}")

    file_size_bytes = os.path.getsize(output_csv)
    print(f"\n5. Output File Size: {file_size_bytes:,} bytes ({file_size_bytes / (1024 * 1024):.2f} MB)")

    print("\n6. First 10 Rows Preview:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_all.head(10))

if __name__ == "__main__":
    build_player_match_dataset()
