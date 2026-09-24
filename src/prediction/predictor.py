import os
import json
import pickle
import numpy as np
import pandas as pd

class CricketPredictor:
    """
    Cricket Player ML Prediction Engine.
    
    Loads trained ML models once and generates leakage-free player performance predictions
    using historical features strictly preceding the prediction point.
    """

    def __init__(self, models_dir=None, data_path=None):
        base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        
        if models_dir is None:
            models_dir = os.path.join(base_dir, "models")
        if data_path is None:
            data_path = os.path.join(base_dir, "data", "processed", "player_match_data.csv")

        self.models_dir = models_dir
        self.data_path = data_path
        
        metadata_file = os.path.join(self.models_dir, "model_metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Model metadata not found at: {metadata_file}")
            
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        # Get target definitions and feature columns
        self.target_map = {
            'target_runs': 'runs_model.pkl',
            'target_batting_average': 'batting_average_model.pkl',
            'target_strike_rate': 'strike_rate_model.pkl',
            'target_balls_per_boundary': 'balls_per_boundary_model.pkl',
            'target_wickets': 'wickets_model.pkl',
            'target_economy_rate': 'economy_rate_model.pkl',
            'target_bowling_strike_rate': 'bowling_strike_rate_model.pkl',
        }
        
        first_target_meta = next(iter(self.metadata.values()))
        self.feature_cols = first_target_meta['feature_columns']
        
        # Load models into memory once
        self.models = {}
        for target_key, pkl_filename in self.target_map.items():
            model_path = os.path.join(self.models_dir, pkl_filename)
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.models[target_key] = pickle.load(f)
            else:
                raise FileNotFoundError(f"Model file missing: {model_path}")
                
        # Load player match history once
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Processed dataset not found at: {self.data_path}")
            
        self.df_pm = pd.read_csv(self.data_path)
        self.df_pm['player'] = self.df_pm['player'].astype(str).str.strip().str.lower()
        self.known_players = set(self.df_pm['player'].unique())

    def reconstruct_features(self, player_history):
        """
        Reconstruct the exact 31 historical features from a player's match history DataFrame.
        """
        n_matches = len(player_history)
        batting_df = player_history[player_history['batting_matches'] == 1]
        bowling_df = player_history[player_history['bowling_matches'] == 1]

        has_batting = len(batting_df) > 0
        has_bowling = len(bowling_df) > 0

        # Career Batting
        prev_bat_m = len(batting_df)
        c_runs = int(batting_df['batting_runs'].sum()) if has_batting else 0
        c_dism = int(batting_df['batting_dismissed'].sum()) if has_batting else 0
        c_balls = int(batting_df['batting_balls_approx'].sum()) if has_batting else 0
        c_bound = int(batting_df['batting_boundaries'].sum()) if has_batting else 0

        c_avg = round(c_runs / c_dism, 2) if c_dism > 0 else float(c_runs)
        c_sr = round((c_runs / c_balls) * 100.0, 2) if c_balls > 0 else 0.0

        # Rolling Batting (last 3, 5, 10 batting matches)
        r_bat_stats = {}
        for n in [3, 5, 10]:
            sub_b = batting_df.tail(n)
            r_r = int(sub_b['batting_runs'].sum()) if not sub_b.empty else 0
            r_d = int(sub_b['batting_dismissed'].sum()) if not sub_b.empty else 0
            r_bl = int(sub_b['batting_balls_approx'].sum()) if not sub_b.empty else 0
            r_bd = int(sub_b['batting_boundaries'].sum()) if not sub_b.empty else 0

            r_bat_stats[f'runs_last_{n}'] = r_r
            r_bat_stats[f'batting_average_last_{n}'] = round(r_r / r_d, 2) if r_d > 0 else float(r_r)
            r_bat_stats[f'strike_rate_last_{n}'] = round((r_r / r_bl) * 100.0, 2) if r_bl > 0 else 0.0

            if n == 5:
                r_bat_stats['boundaries_last_5'] = r_bd
                r_bat_stats['balls_per_boundary_last_5'] = round(r_bl / r_bd, 2) if (has_batting and r_bd > 0) else np.nan

        # Career Bowling
        prev_bowl_m = len(bowling_df)
        c_wick = int(bowling_df['bowling_wickets_approx'].sum()) if has_bowling else 0
        c_bruns = int(bowling_df['bowling_runs_conceded_approx'].sum()) if has_bowling else 0
        c_deliv = int(bowling_df['bowling_deliveries'].sum()) if has_bowling else 0

        c_econ = round(c_bruns / (c_deliv / 6.0), 2) if c_deliv > 0 else 0.0
        c_bsr = round(c_deliv / c_wick, 2) if c_wick > 0 else np.nan

        # Rolling Bowling (last 3, 5, 10 bowling matches)
        r_bowl_stats = {}
        for n in [3, 5, 10]:
            sub_w = bowling_df.tail(n)
            r_wk = int(sub_w['bowling_wickets_approx'].sum()) if not sub_w.empty else 0
            r_rc = int(sub_w['bowling_runs_conceded_approx'].sum()) if not sub_w.empty else 0
            r_dv = int(sub_w['bowling_deliveries'].sum()) if not sub_w.empty else 0

            r_bowl_stats[f'wickets_last_{n}'] = r_wk
            r_bowl_stats[f'economy_last_{n}'] = round(r_rc / (r_dv / 6.0), 2) if r_dv > 0 else 0.0

            if n == 5:
                r_bowl_stats['bowling_strike_rate_last_5'] = round(r_dv / r_wk, 2) if (has_bowling and r_wk > 0) else np.nan
            if n == 10:
                r_bowl_stats['bowling_strike_rate_last_10'] = round(r_dv / r_wk, 2) if (has_bowling and r_wk > 0) else np.nan

        feat_dict = {
            'previous_matches': n_matches,
            'previous_batting_matches': prev_bat_m,
            'career_runs_before_match': c_runs,
            'career_dismissals_before_match': c_dism,
            'career_average_before_match': c_avg,
            'career_strike_rate_before_match': c_sr,
            'career_boundaries_before_match': c_bound,
            'runs_last_3': r_bat_stats['runs_last_3'],
            'runs_last_5': r_bat_stats['runs_last_5'],
            'runs_last_10': r_bat_stats['runs_last_10'],
            'batting_average_last_3': r_bat_stats['batting_average_last_3'],
            'batting_average_last_5': r_bat_stats['batting_average_last_5'],
            'batting_average_last_10': r_bat_stats['batting_average_last_10'],
            'strike_rate_last_3': r_bat_stats['strike_rate_last_3'],
            'strike_rate_last_5': r_bat_stats['strike_rate_last_5'],
            'strike_rate_last_10': r_bat_stats['strike_rate_last_10'],
            'boundaries_last_5': r_bat_stats['boundaries_last_5'],
            'balls_per_boundary_last_5': r_bat_stats['balls_per_boundary_last_5'],
            'previous_bowling_matches': prev_bowl_m,
            'career_wickets_before_match': c_wick,
            'career_bowling_runs_before_match': c_bruns,
            'career_economy_before_match': c_econ,
            'career_bowling_strike_rate_before_match': c_bsr,
            'wickets_last_3': r_bowl_stats['wickets_last_3'],
            'wickets_last_5': r_bowl_stats['wickets_last_5'],
            'wickets_last_10': r_bowl_stats['wickets_last_10'],
            'economy_last_3': r_bowl_stats['economy_last_3'],
            'economy_last_5': r_bowl_stats['economy_last_5'],
            'economy_last_10': r_bowl_stats['economy_last_10'],
            'bowling_strike_rate_last_5': r_bowl_stats['bowling_strike_rate_last_5'],
            'bowling_strike_rate_last_10': r_bowl_stats['bowling_strike_rate_last_10'],
        }

        # Build 1-row DataFrame & verify strict feature column alignment
        X = pd.DataFrame([feat_dict])
        missing_cols = [c for c in self.feature_cols if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Feature reconstruction mismatch. Missing required feature columns: {missing_cols}")
            
        X = X[self.feature_cols]
        return X, has_batting, has_bowling

    def predict_player_for_match(self, player_name, match_id=None):
        """
        Generates leakage-free predictions for a player.
        
        If match_id is provided, only matches occurring strictly BEFORE match_id are used.
        If match_id is None, all available historical data up to the latest match is used.
        """
        clean_name = str(player_name).strip().lower()
        if clean_name not in self.known_players:
            return {
                "player": player_name,
                "status": "error",
                "message": f"Player '{player_name}' not found in historical records."
            }

        player_history = self.df_pm[self.df_pm['player'] == clean_name]
        
        if match_id is not None:
            # Filter strictly BEFORE match_id (never include current match_id or future matches)
            player_history = player_history[player_history['match_id'] < match_id]

        if player_history.empty:
            return {
                "player": player_name,
                "status": "insufficient_history",
                "message": f"No prior historical match data available for player '{player_name}' before match_id {match_id}."
            }

        prediction_point_m_id = player_history['match_id'].max()
        X, has_batting, has_bowling = self.reconstruct_features(player_history)

        # Generate predictions per target
        raw_preds = {}
        for target_key, model in self.models.items():
            pred_val = model.predict(X)[0]
            raw_preds[target_key] = float(pred_val)

        # Apply post-processing non-negativity bounds
        res = {
            "player": player_name,
            "status": "success",
            "prediction_point_match_id": int(prediction_point_m_id),
            "historical_matches_used": len(player_history),
            "has_batting_history": has_batting,
            "has_bowling_history": has_bowling,
            "uncertainty_available": False,
        }

        if has_batting:
            res["batting"] = {
                "expected_runs": round(max(0.0, raw_preds['target_runs']), 1),
                "x_batting_average": round(max(0.0, raw_preds['target_batting_average']), 1),
                "x_strike_rate": round(max(0.0, raw_preds['target_strike_rate']), 1),
                "x_balls_per_boundary": round(max(0.0, raw_preds['target_balls_per_boundary']), 1),
            }
        else:
            res["batting"] = {
                "status": "insufficient_history",
                "message": "Player has no prior batting history."
            }

        if has_bowling:
            res["bowling"] = {
                "x_wickets": round(max(0.0, raw_preds['target_wickets']), 1),
                "x_economy_rate": round(max(0.0, raw_preds['target_economy_rate']), 1),
                "x_bowling_strike_rate": round(max(0.0, raw_preds['target_bowling_strike_rate']), 1),
            }
        else:
            res["bowling"] = {
                "status": "insufficient_history",
                "message": "Player has no prior bowling history."
            }

        return res

    def predict_player(self, player_name):
        """
        Alias for predict_player_for_match(player_name, match_id=None).
        """
        return self.predict_player_for_match(player_name, match_id=None)

    def predict_players(self, player_names):
        """
        Batch prediction for a list of player names.
        """
        return [self.predict_player(p) for p in player_names]

    def predict_all_known_players(self, min_matches=1):
        """
        Batch predictions for all known players with sufficient history.
        """
        print(f"Generating batch predictions for all known players (min_matches >= {min_matches})...")
        results = []
        for p in self.known_players:
            res = self.predict_player(p)
            if res.get("status") == "success" and res.get("historical_matches_used", 0) >= min_matches:
                results.append(res)
        return results
