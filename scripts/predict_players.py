import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.prediction.predictor import CricketPredictor

def format_prediction_output(pred):
    player = pred.get("player", "Unknown")
    status = pred.get("status")

    if status == "error":
        print(f"\n[ERROR] {pred.get('message', 'Player not found')}\n")
        return

    print("=" * 60)
    print(f"PLAYER: {player.upper()}")
    print("=" * 60)
    print(f"Prediction Point (Last Match ID): {pred.get('prediction_point_match_id')}")
    print(f"Historical Matches Evaluated:     {pred.get('historical_matches_used')}")
    print("-" * 60)

    # Batting Section
    batting = pred.get("batting", {})
    if pred.get("has_batting_history") and isinstance(batting, dict) and "expected_runs" in batting:
        print("BATTING PREDICTIONS:")
        print(f"  Expected Runs:        {batting.get('expected_runs'):>8.1f}")
        print(f"  x Batting Average:    {batting.get('x_batting_average'):>8.1f}")
        print(f"  x Strike Rate:        {batting.get('x_strike_rate'):>8.1f}")
        bpb = batting.get('x_balls_per_boundary')
        bpb_str = f"{bpb:>8.1f}" if bpb is not None else "     N/A"
        print(f"  x Balls / Boundary:   {bpb_str}")
    else:
        print("BATTING PREDICTIONS:")
        print("  Insufficient historical batting data available.")

    print("-" * 60)

    # Bowling Section
    bowling = pred.get("bowling", {})
    if pred.get("has_bowling_history") and isinstance(bowling, dict) and "x_wickets" in bowling:
        print("BOWLING PREDICTIONS:")
        print(f"  x Wickets:            {bowling.get('x_wickets'):>8.1f}")
        print(f"  x Economy Rate:       {bowling.get('x_economy_rate'):>8.1f}")
        bsr = bowling.get('x_bowling_strike_rate')
        bsr_str = f"{bsr:>8.1f}" if bsr is not None else "     N/A"
        print(f"  x Bowling Strike Rate:{bsr_str}")
    else:
        print("BOWLING PREDICTIONS:")
        print("  Insufficient historical bowling data available.")

    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Cricket Player ML Prediction CLI")
    parser.add_argument("--player", type=str, help="Name of the player to generate predictions for")
    parser.add_argument("--all", action="store_true", help="Generate predictions for all known players with sufficient history")
    parser.add_argument("--min-matches", type=int, default=5, help="Minimum matches threshold when using --all (default: 5)")
    args = parser.parse_args()

    if not args.player and not args.all:
        parser.print_help()
        sys.exit(1)

    print("Initializing Cricket ML Prediction Engine...")
    predictor = CricketPredictor()

    if args.player:
        pred = predictor.predict_player(args.player)
        format_prediction_output(pred)

    elif args.all:
        print(f"\nGenerating batch predictions for all players with >= {args.min_matches} matches...")
        all_preds = predictor.predict_all_known_players(min_matches=args.min_matches)
        print(f"Successfully generated predictions for {len(all_preds):,} players.")
        print("Showing preview of first 5 players:\n")
        for p in all_preds[:5]:
            format_prediction_output(p)

if __name__ == "__main__":
    main()
