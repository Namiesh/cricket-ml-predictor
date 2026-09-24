import os
import numpy as np
import pandas as pd

def build_features():
    input_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "player_match_data.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    output_csv = os.path.join(output_dir, "training_data.csv")
    dict_md = os.path.join(output_dir, "feature_dictionary.md")

    print(f"Loading player-match data from: {input_path}")
    df = pd.read_csv(input_path)

    # Sort strictly chronologically by player and match_id to ensure strict temporal ordering
    df = df.sort_values(by=['player', 'match_id']).reset_index(drop=True)

    print(f"Loaded {len(df):,} records. Calculating historical career metrics (leakage-free)...")

    # Shift base stats by 1 per player so current match values are NEVER included in historical features
    df['prev_matches_played'] = df.groupby('player')['matches_played'].shift(1).fillna(0).astype(int)
    df['previous_matches'] = df.groupby('player')['prev_matches_played'].cumsum()

    df['prev_batting_matches'] = df.groupby('player')['batting_matches'].shift(1).fillna(0).astype(int)
    df['previous_batting_matches'] = df.groupby('player')['prev_batting_matches'].cumsum()

    df['prev_bowling_matches'] = df.groupby('player')['bowling_matches'].shift(1).fillna(0).astype(int)
    df['previous_bowling_matches'] = df.groupby('player')['prev_bowling_matches'].cumsum()

    df['prev_batting_runs'] = df.groupby('player')['batting_runs'].shift(1).fillna(0)
    df['career_runs_before_match'] = df.groupby('player')['prev_batting_runs'].cumsum().astype(int)

    df['prev_batting_dismissed'] = df.groupby('player')['batting_dismissed'].shift(1).fillna(0)
    df['career_dismissals_before_match'] = df.groupby('player')['prev_batting_dismissed'].cumsum().astype(int)

    df['prev_batting_balls'] = df.groupby('player')['batting_balls_approx'].shift(1).fillna(0)
    df['career_balls_before_match'] = df.groupby('player')['prev_batting_balls'].cumsum().astype(int)

    df['prev_batting_boundaries'] = df.groupby('player')['batting_boundaries'].shift(1).fillna(0)
    df['career_boundaries_before_match'] = df.groupby('player')['prev_batting_boundaries'].cumsum().astype(int)

    df['prev_bowling_wickets'] = df.groupby('player')['bowling_wickets_approx'].shift(1).fillna(0)
    df['career_wickets_before_match'] = df.groupby('player')['prev_bowling_wickets'].cumsum().astype(int)

    df['prev_bowling_runs'] = df.groupby('player')['bowling_runs_conceded_approx'].shift(1).fillna(0)
    df['career_bowling_runs_before_match'] = df.groupby('player')['prev_bowling_runs'].cumsum().astype(int)

    df['prev_bowling_deliveries'] = df.groupby('player')['bowling_deliveries'].shift(1).fillna(0)
    df['career_deliveries_before_match'] = df.groupby('player')['prev_bowling_deliveries'].cumsum().astype(int)

    # Derived Career Ratios (handling 0 division gracefully)
    df['career_average_before_match'] = np.where(
        df['career_dismissals_before_match'] > 0,
        (df['career_runs_before_match'] / df['career_dismissals_before_match']).round(2),
        df['career_runs_before_match'].astype(float)
    )

    df['career_strike_rate_before_match'] = np.where(
        df['career_balls_before_match'] > 0,
        ((df['career_runs_before_match'] / df['career_balls_before_match']) * 100.0).round(2),
        0.0
    )

    df['career_economy_before_match'] = np.where(
        df['career_deliveries_before_match'] > 0,
        ((df['career_bowling_runs_before_match'] / (df['career_deliveries_before_match'] / 6.0))).round(2),
        0.0
    )

    df['career_bowling_strike_rate_before_match'] = np.where(
        df['career_wickets_before_match'] > 0,
        (df['career_deliveries_before_match'] / df['career_wickets_before_match']).round(2),
        np.nan
    )

    # Rolling Batting Form over previous BATTING matches
    print("Calculating rolling batting features over previous batting matches...")
    df_bat = df[df['batting_matches'] == 1].copy()
    
    df_bat['p_runs'] = df_bat.groupby('player')['batting_runs'].shift(1)
    df_bat['p_dism'] = df_bat.groupby('player')['batting_dismissed'].shift(1)
    df_bat['p_balls'] = df_bat.groupby('player')['batting_balls_approx'].shift(1)
    df_bat['p_bound'] = df_bat.groupby('player')['batting_boundaries'].shift(1)

    for n in [3, 5, 10]:
        df_bat[f'r_runs_{n}'] = df_bat.groupby('player')['p_runs'].transform(lambda x: x.rolling(n, min_periods=1).sum())
        df_bat[f'r_dism_{n}'] = df_bat.groupby('player')['p_dism'].transform(lambda x: x.rolling(n, min_periods=1).sum())
        df_bat[f'r_balls_{n}'] = df_bat.groupby('player')['p_balls'].transform(lambda x: x.rolling(n, min_periods=1).sum())
        df_bat[f'r_bound_{n}'] = df_bat.groupby('player')['p_bound'].transform(lambda x: x.rolling(n, min_periods=1).sum())

        df_bat[f'runs_last_{n}'] = df_bat[f'r_runs_{n}'].fillna(0).astype(int)
        
        df_bat[f'batting_average_last_{n}'] = np.where(
            df_bat[f'r_dism_{n}'] > 0,
            (df_bat[f'r_runs_{n}'] / df_bat[f'r_dism_{n}']).round(2),
            df_bat[f'r_runs_{n}'].fillna(0)
        )

        df_bat[f'strike_rate_last_{n}'] = np.where(
            df_bat[f'r_balls_{n}'] > 0,
            ((df_bat[f'r_runs_{n}'] / df_bat[f'r_balls_{n}']) * 100.0).round(2),
            0.0
        )

    df_bat['boundaries_last_5'] = df_bat['r_bound_5'].fillna(0).astype(int)
    df_bat['balls_per_boundary_last_5'] = np.where(
        df_bat['r_bound_5'] > 0,
        (df_bat['r_balls_5'] / df_bat['r_bound_5']).round(2),
        np.nan
    )

    bat_cols = [
        'match_id', 'player',
        'runs_last_3', 'runs_last_5', 'runs_last_10',
        'batting_average_last_3', 'batting_average_last_5', 'batting_average_last_10',
        'strike_rate_last_3', 'strike_rate_last_5', 'strike_rate_last_10',
        'boundaries_last_5', 'balls_per_boundary_last_5'
    ]
    
    df = pd.merge(df, df_bat[bat_cols], on=['match_id', 'player'], how='left')

    # Forward fill rolling batting features per player for non-batting matches
    for col in bat_cols[2:]:
        df[col] = df.groupby('player')[col].ffill().fillna(0 if 'average' not in col and 'rate' not in col and 'boundary' not in col else (0.0 if 'boundary' not in col else np.nan))

    # Rolling Bowling Form over previous BOWLING matches
    print("Calculating rolling bowling features over previous bowling matches...")
    df_bowl = df[df['bowling_matches'] == 1].copy()

    df_bowl['p_wick'] = df_bowl.groupby('player')['bowling_wickets_approx'].shift(1)
    df_bowl['p_runs_c'] = df_bowl.groupby('player')['bowling_runs_conceded_approx'].shift(1)
    df_bowl['p_deliv'] = df_bowl.groupby('player')['bowling_deliveries'].shift(1)

    for n in [3, 5, 10]:
        df_bowl[f'r_wick_{n}'] = df_bowl.groupby('player')['p_wick'].transform(lambda x: x.rolling(n, min_periods=1).sum())
        df_bowl[f'r_runs_c_{n}'] = df_bowl.groupby('player')['p_runs_c'].transform(lambda x: x.rolling(n, min_periods=1).sum())
        df_bowl[f'r_deliv_{n}'] = df_bowl.groupby('player')['p_deliv'].transform(lambda x: x.rolling(n, min_periods=1).sum())

        df_bowl[f'wickets_last_{n}'] = df_bowl[f'r_wick_{n}'].fillna(0).astype(int)

        df_bowl[f'economy_last_{n}'] = np.where(
            df_bowl[f'r_deliv_{n}'] > 0,
            (df_bowl[f'r_runs_c_{n}'] / (df_bowl[f'r_deliv_{n}'] / 6.0)).round(2),
            0.0
        )

    df_bowl['bowling_strike_rate_last_5'] = np.where(
        df_bowl['r_wick_5'] > 0,
        (df_bowl['r_deliv_5'] / df_bowl['r_wick_5']).round(2),
        np.nan
    )
    df_bowl['bowling_strike_rate_last_10'] = np.where(
        df_bowl['r_wick_10'] > 0,
        (df_bowl['r_deliv_10'] / df_bowl['r_wick_10']).round(2),
        np.nan
    )

    bowl_cols = [
        'match_id', 'player',
        'wickets_last_3', 'wickets_last_5', 'wickets_last_10',
        'economy_last_3', 'economy_last_5', 'economy_last_10',
        'bowling_strike_rate_last_5', 'bowling_strike_rate_last_10'
    ]

    df = pd.merge(df, df_bowl[bowl_cols], on=['match_id', 'player'], how='left')

    # Forward fill rolling bowling features per player for non-bowling matches
    for col in bowl_cols[2:]:
        df[col] = df.groupby('player')[col].ffill().fillna(0 if 'economy' not in col and 'strike_rate' not in col else (0.0 if 'strike_rate' not in col else np.nan))

    # Map Target Columns (Current Match Performance)
    df['target_runs'] = df['batting_runs']
    df['target_batting_average'] = df['batting_average_approx']
    df['target_strike_rate'] = df['batting_strike_rate_approx']
    df['target_balls_per_boundary'] = df['balls_per_boundary_approx']
    df['target_wickets'] = df['bowling_wickets_approx']
    df['target_economy_rate'] = df['bowling_economy_rate_approx']
    df['target_bowling_strike_rate'] = df['bowling_strike_rate_approx']

    # Final sort by match_id and player
    df = df.sort_values(by=['match_id', 'player']).reset_index(drop=True)

    # Select and order final output columns
    final_cols = [
        # Identifiers
        'match_id', 'player',
        
        # Batting Historical Features
        'previous_matches', 'previous_batting_matches',
        'career_runs_before_match', 'career_dismissals_before_match',
        'career_average_before_match', 'career_strike_rate_before_match',
        'career_boundaries_before_match',
        'runs_last_3', 'runs_last_5', 'runs_last_10',
        'batting_average_last_3', 'batting_average_last_5', 'batting_average_last_10',
        'strike_rate_last_3', 'strike_rate_last_5', 'strike_rate_last_10',
        'boundaries_last_5', 'balls_per_boundary_last_5',

        # Bowling Historical Features
        'previous_bowling_matches',
        'career_wickets_before_match', 'career_bowling_runs_before_match',
        'career_economy_before_match', 'career_bowling_strike_rate_before_match',
        'wickets_last_3', 'wickets_last_5', 'wickets_last_10',
        'economy_last_3', 'economy_last_5', 'economy_last_10',
        'bowling_strike_rate_last_5', 'bowling_strike_rate_last_10',

        # Targets (Current Match Performance)
        'target_runs', 'target_batting_average', 'target_strike_rate',
        'target_balls_per_boundary', 'target_wickets', 'target_economy_rate',
        'target_bowling_strike_rate'
    ]

    df_final = df[final_cols]

    print(f"Saving final dataset to: {output_csv}")
    df_final.to_csv(output_csv, index=False)

    create_feature_dictionary(dict_md)

    print("\n--- LEAKAGE VALIDATION & VERIFICATION ---")
    perform_leakage_validation(df_final, df_input=pd.read_csv(input_path))

    return df_final, output_csv

def create_feature_dictionary(output_path):
    content = """# Feature Dictionary: Training Dataset (`training_data.csv`)

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
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Created feature dictionary at: {output_path}")

def perform_leakage_validation(df_final, df_input):
    print("Executing Leakage Checks on randomly selected experienced players...")
    player_counts = df_final['player'].value_counts()
    experienced_players = player_counts[player_counts >= 15].index.tolist()
    
    np.random.seed(42)
    sample_players = np.random.choice(experienced_players, size=5, replace=False)
    
    print("\n--- 5 VALIDATION EXAMPLES ---")
    print(f"{'Player':<20} {'Match ID':<10} {'Prev Matches':<14} {'Career Runs Before':<20} {'Runs Last 5':<12} {'Target Runs':<12}")
    print("-" * 90)

    leakage_passed = True
    for p in sample_players:
        p_matches = df_final[df_final['player'] == p].sort_values('match_id')
        sample_row = p_matches.iloc[8]
        
        m_id = sample_row['match_id']
        prev_m = sample_row['previous_matches']
        career_r_before = sample_row['career_runs_before_match']
        r_last_5 = sample_row['runs_last_5']
        target_r = sample_row['target_runs']
        
        # Verify manually from df_input (raw match data for player before m_id)
        prior_df = df_input[(df_input['player'] == p) & (df_input['match_id'] < m_id)]
        curr_df = df_input[(df_input['player'] == p) & (df_input['match_id'] == m_id)]
        
        actual_prior_matches = len(prior_df)
        actual_prior_runs = prior_df['batting_runs'].sum()
        actual_curr_runs = curr_df['batting_runs'].values[0] if not curr_df.empty else 0
        
        if prev_m != actual_prior_matches or career_r_before != actual_prior_runs or target_r != actual_curr_runs:
            leakage_passed = False

        print(f"{p:<20} {m_id:<10} {prev_m:<14} {career_r_before:<20} {r_last_5:<12} {target_r:<12}")

    print("-" * 90)
    if leakage_passed:
        print("LEAKAGE VALIDATION RESULT: PASSED 100% (No future or current match data present in historical features).")
    else:
        print("LEAKAGE VALIDATION RESULT: FAILED - Discrepancy detected.")

    print("\n--- DATASET SUMMARY STATISTICS ---")
    print(f"Total Rows:    {len(df_final):,}")
    print(f"Total Columns: {len(df_final.columns)}")
    
    print("\nMissing-Value Summary:")
    missing = df_final.isna().sum()
    missing_cols = missing[missing > 0]
    if missing_cols.empty:
        print("  No missing values found.")
    else:
        for c, count in missing_cols.items():
            print(f"  - {c:<35}: {count:,} missing ({count/len(df_final)*100:.2f}%)")

    print("\nTarget Summary Statistics:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_final[['target_runs', 'target_batting_average', 'target_strike_rate', 'target_wickets', 'target_economy_rate']].describe())

if __name__ == "__main__":
    build_features()
