import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

def train_all_models():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "training_data.csv")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    print(f"Loading training data from: {data_path}")
    df = pd.read_csv(data_path)

    # Automatically identify feature columns
    target_cols = [c for c in df.columns if c.startswith('target_')]
    exclude_cols = set(target_cols + ['match_id', 'player'])
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"Identified {len(feature_cols)} feature columns:")
    for c in feature_cols:
        print(f"  - {c}")

    # Validation Checks
    assert not any(c.startswith('target_') for c in feature_cols), "Validation Error: target column found in features!"
    assert 'match_id' not in feature_cols, "Validation Error: match_id found in features!"
    assert 'player' not in feature_cols, "Validation Error: player column found in features!"

    # Chronological match_id split (First 80% matches -> Train, Last 20% -> Test)
    unique_matches = sorted(df['match_id'].unique())
    split_idx = int(len(unique_matches) * 0.8)
    train_matches = set(unique_matches[:split_idx])
    test_matches = set(unique_matches[split_idx:])

    min_train_m, max_train_m = min(train_matches), max(train_matches)
    min_test_m, max_test_m = min(test_matches), max(test_matches)

    print("\n--- TRAIN / TEST MATCH_ID SPLIT ---")
    print(f"Total Matches: {len(unique_matches):,}")
    print(f"Train Matches: {len(train_matches):,} (match_id range: {min_train_m} to {max_train_m})")
    print(f"Test Matches:  {len(test_matches):,} (match_id range: {min_test_m} to {max_test_m})")
    
    # Validation assertion: Test set strictly succeeds train set
    assert max_train_m < min_test_m, "Leakage Error: Test matches overlap with or precede train matches!"
    print("Validation Passed: Test match_ids strictly succeed train match_ids.")

    targets_to_process = [
        ('target_runs', 'runs_model.pkl'),
        ('target_batting_average', 'batting_average_model.pkl'),
        ('target_strike_rate', 'strike_rate_model.pkl'),
        ('target_balls_per_boundary', 'balls_per_boundary_model.pkl'),
        ('target_wickets', 'wickets_model.pkl'),
        ('target_economy_rate', 'economy_rate_model.pkl'),
        ('target_bowling_strike_rate', 'bowling_strike_rate_model.pkl'),
    ]

    all_results = []
    metadata = {}
    trained_models = []
    skipped_models = []

    print("\n--- MODEL TRAINING & EVALUATION ---")
    for target_col, model_filename in targets_to_process:
        print(f"\nProcessing Target: {target_col}")

        # Filter non-NaN target observations
        valid_df = df[df[target_col].notna()].copy()
        
        valid_train = valid_df[valid_df['match_id'].isin(train_matches)]
        valid_test = valid_df[valid_df['match_id'].isin(test_matches)]

        n_train = len(valid_train)
        n_test = len(valid_test)

        if n_train < 1000 or n_test < 100:
            print(f"SKIPPING {target_col}: Insufficient valid training/test samples (train={n_train}, test={n_test}).")
            skipped_models.append({
                'target': target_col,
                'reason': f"Insufficient valid observations (train={n_train}, test={n_test})"
            })
            continue

        X_train = valid_train[feature_cols]
        y_train = valid_train[target_col]
        X_test = valid_test[feature_cols]
        y_test = valid_test[target_col]

        # Verify training and test feature columns are identical
        assert list(X_train.columns) == list(X_test.columns), "Feature mismatch between train and test!"

        # Baseline Prediction (Mean training target)
        y_train_mean = float(y_train.mean())
        baseline_preds = np.full_like(y_test, fill_value=y_train_mean)
        baseline_mae = mean_absolute_error(y_test, baseline_preds)

        # HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            random_state=42,
            max_iter=100,
            learning_rate=0.1
        )
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        medae = median_absolute_error(y_test, preds)

        print(f"  Train Samples: {n_train:,} | Test Samples: {n_test:,}")
        print(f"  ML MAE: {mae:.4f} | Baseline MAE: {baseline_mae:.4f} | Imprv: {baseline_mae - mae:.4f}")
        print(f"  RMSE: {rmse:.4f} | R²: {r2:.4f} | MedAE: {medae:.4f}")

        # Save model pkl
        model_save_path = os.path.join(models_dir, model_filename)
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved model to: {model_save_path}")

        result_row = {
            'target': target_col,
            'train_samples': n_train,
            'test_samples': n_test,
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'r2': round(r2, 4),
            'medae': round(medae, 4),
            'baseline_mae': round(baseline_mae, 4)
        }
        all_results.append(result_row)
        trained_models.append(target_col)

        metadata[target_col] = {
            'model_name': 'HistGradientBoostingRegressor',
            'model_filename': model_filename,
            'target': target_col,
            'feature_columns': feature_cols,
            'train_samples': n_train,
            'test_samples': n_test,
            'mae': round(float(mae), 4),
            'rmse': round(float(rmse), 4),
            'r2': round(float(r2), 4),
            'medae': round(float(medae), 4),
            'baseline_mae': round(float(baseline_mae), 4),
            'train_match_id_range': [int(min_train_m), int(max_train_m)],
            'test_match_id_range': [int(min_test_m), int(max_test_m)]
        }

    # Save metadata JSON
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata JSON to: {metadata_path}")

    # Results Table
    res_df = pd.DataFrame(all_results)
    print("\n" + "=" * 105)
    print("COMPACT MODEL EVALUATION RESULTS TABLE")
    print("=" * 105)
    print(f"{'Target':<28} {'Train Samples':<15} {'Test Samples':<15} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'Baseline MAE':<12}")
    print("-" * 105)
    for _, r in res_df.iterrows():
        print(f"{r['target']:<28} {r['train_samples']:<15,} {r['test_samples']:<15,} {r['mae']:<10.4f} {r['rmse']:<10.4f} {r['r2']:<10.4f} {r['baseline_mae']:<12.4f}")
    print("=" * 105)

if __name__ == "__main__":
    train_all_models()
