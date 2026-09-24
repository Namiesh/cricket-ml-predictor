"""
Evaluate trained cricket prediction models.

Calculates:
- MAE
- RMSE
- R²
- Mean baseline MAE
- Percentage of predictions within ±5
- Percentage within ±10
- Percentage within ±20

Uses the same chronological 80/20 train-test split
as the model training pipeline.
"""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "training_data.csv"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "model_evaluation.csv"


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

MODEL_CONFIG = {
    "target_runs": {
        "model": "runs_model.pkl",
        "display_name": "Runs",
    },
    "target_batting_average": {
        "model": "batting_average_model.pkl",
        "display_name": "Batting Average",
    },
    "target_strike_rate": {
        "model": "strike_rate_model.pkl",
        "display_name": "Strike Rate",
    },
    "target_balls_per_boundary": {
        "model": "balls_per_boundary_model.pkl",
        "display_name": "Balls per Boundary",
    },
    "target_wickets": {
        "model": "wickets_model.pkl",
        "display_name": "Wickets",
    },
    "target_economy_rate": {
        "model": "economy_rate_model.pkl",
        "display_name": "Economy Rate",
    },
    "target_bowling_strike_rate": {
        "model": "bowling_strike_rate_model.pkl",
        "display_name": "Bowling Strike Rate",
    },
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def load_model(path):
    """Load a trained sklearn model."""
    with open(path, "rb") as f:
        return pickle.load(f)


def tolerance_accuracy(y_true, y_pred, tolerance):
    """
    Percentage of predictions whose absolute error
    is within the specified tolerance.
    """
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return np.mean(errors <= tolerance) * 100


def get_feature_columns(df):
    """
    Get the numerical feature columns used by the models.

    The feature pipeline created 31 historical features.
    Targets and identifiers are excluded.
    """

    excluded_columns = {
        "match_id",
        "player",

        # Targets
        "target_runs",
        "target_batting_average",
        "target_strike_rate",
        "target_balls_per_boundary",
        "target_wickets",
        "target_economy_rate",
        "target_bowling_strike_rate",

        # Non-feature metadata
        "striker",
        "bowler",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    return feature_columns


def chronological_split(df):
    """
    Reproduce the chronological split used during training.

    First 80% of unique match IDs -> training
    Last 20% -> testing
    """

    match_ids = np.sort(df["match_id"].unique())

    split_index = int(len(match_ids) * 0.80)

    train_match_ids = set(match_ids[:split_index])
    test_match_ids = set(match_ids[split_index:])

    train_df = df[df["match_id"].isin(train_match_ids)].copy()
    test_df = df[df["match_id"].isin(test_match_ids)].copy()

    return train_df, test_df


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("CRICKET ML MODEL EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{DATA_PATH}"
        )

    print(f"\nLoading dataset:")
    print(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    print(f"Total rows: {len(df):,}")

    # -------------------------------------------------------------
    # Feature columns
    # -------------------------------------------------------------

    feature_columns = get_feature_columns(df)

    print(f"\nFeatures detected: {len(feature_columns)}")

    # -------------------------------------------------------------
    # Chronological split
    # -------------------------------------------------------------

    train_df, test_df = chronological_split(df)

    print("\nChronological split:")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows : {len(test_df):,}")

    print(
        f"Training matches: "
        f"{train_df['match_id'].min()} → {train_df['match_id'].max()}"
    )

    print(
        f"Testing matches : "
        f"{test_df['match_id'].min()} → {test_df['match_id'].max()}"
    )

    # -------------------------------------------------------------
    # Evaluate each model
    # -------------------------------------------------------------

    results = []

    for target, config in MODEL_CONFIG.items():

        model_path = MODELS_DIR / config["model"]

        print("\n" + "-" * 70)
        print(f"Evaluating: {config['display_name']}")
        print(f"Target: {target}")
        print(f"Model: {model_path.name}")

        if not model_path.exists():
            print("WARNING: Model file not found. Skipping.")
            continue

        # ---------------------------------------------------------
        # Remove rows where target is unavailable
        # ---------------------------------------------------------

        train_valid = train_df.dropna(
            subset=[target] + feature_columns
        )

        test_valid = test_df.dropna(
            subset=[target] + feature_columns
        )

        if len(test_valid) == 0:
            print("WARNING: No valid test samples. Skipping.")
            continue

        X_test = test_valid[feature_columns]
        y_test = test_valid[target]

        # ---------------------------------------------------------
        # Load model
        # ---------------------------------------------------------

        model = load_model(model_path)

        # ---------------------------------------------------------
        # Predict
        # ---------------------------------------------------------

        y_pred = model.predict(X_test)

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------

        mae = mean_absolute_error(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        r2 = r2_score(y_test, y_pred)

        # Mean baseline
        baseline_prediction = train_valid[target].mean()

        baseline_pred = np.full(
            len(y_test),
            baseline_prediction
        )

        baseline_mae = mean_absolute_error(
            y_test,
            baseline_pred
        )

        # Improvement over baseline
        baseline_improvement = (
            (baseline_mae - mae) / baseline_mae
        ) * 100

        # Tolerance accuracies
        within_5 = tolerance_accuracy(
            y_test,
            y_pred,
            5
        )

        within_10 = tolerance_accuracy(
            y_test,
            y_pred,
            10
        )

        within_20 = tolerance_accuracy(
            y_test,
            y_pred,
            20
        )

        result = {
            "target": target,
            "display_name": config["display_name"],
            "test_samples": len(test_valid),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "baseline_MAE": baseline_mae,
            "improvement_over_baseline_%": baseline_improvement,
            "within_5_%": within_5,
            "within_10_%": within_10,
            "within_20_%": within_20,
        }

        results.append(result)

        # ---------------------------------------------------------
        # Console output
        # ---------------------------------------------------------

        print(f"Test samples              : {len(test_valid):,}")
        print(f"MAE                       : {mae:.4f}")
        print(f"RMSE                      : {rmse:.4f}")
        print(f"R²                        : {r2:.4f}")
        print(f"Baseline MAE              : {baseline_mae:.4f}")
        print(
            f"Improvement over baseline : "
            f"{baseline_improvement:.2f}%"
        )

        print(f"Within ±5                 : {within_5:.2f}%")
        print(f"Within ±10                : {within_10:.2f}%")
        print(f"Within ±20                : {within_20:.2f}%")

    # -------------------------------------------------------------
    # Create results dataframe
    # -------------------------------------------------------------

    if not results:
        print("\nNo models were evaluated.")
        return

    results_df = pd.DataFrame(results)

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # -------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)

    display_columns = [
        "display_name",
        "MAE",
        "RMSE",
        "R2",
        "within_5_%",
        "within_10_%",
        "within_20_%",
    ]

    summary = results_df[display_columns].copy()

    summary.columns = [
        "Target",
        "MAE",
        "RMSE",
        "R²",
        "±5 Accuracy %",
        "±10 Accuracy %",
        "±20 Accuracy %",
    ]

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\n")
    print(f"Full evaluation saved to:")
    print(OUTPUT_PATH)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()