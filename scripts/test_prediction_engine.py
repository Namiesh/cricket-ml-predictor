import os
import sys
import unittest
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prediction.predictor import CricketPredictor

class TestCricketPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nInitializing CricketPredictor for test suite...")
        cls.predictor = CricketPredictor()

    def test_1_models_loaded_successfully(self):
        """Test 1: Verify all 7 target models load successfully."""
        self.assertEqual(len(self.predictor.models), 7)
        for target in self.predictor.target_map.keys():
            self.assertIn(target, self.predictor.models)
            self.assertIsNotNone(self.predictor.models[target])

    def test_2_required_feature_columns_exist(self):
        """Test 2: Verify required 31 feature columns exist and align."""
        self.assertEqual(len(self.predictor.feature_cols), 31)
        self.assertNotIn("match_id", self.predictor.feature_cols)
        self.assertNotIn("player", self.predictor.feature_cols)
        self.assertFalse(any(c.startswith("target_") for c in self.predictor.feature_cols))

    def test_3_predictions_are_numeric(self):
        """Test 3: Verify output predictions are numeric values."""
        res = self.predictor.predict_player("da warner")
        self.assertEqual(res["status"], "success")
        batting = res["batting"]
        self.assertIsInstance(batting["expected_runs"], (int, float))
        self.assertIsInstance(batting["x_batting_average"], (int, float))
        self.assertIsInstance(batting["x_strike_rate"], (int, float))

    def test_4_predictions_not_nan_or_inf(self):
        """Test 4: Verify predictions contain no NaN or Infinity."""
        res = self.predictor.predict_player("da warner")
        batting = res["batting"]
        for val in batting.values():
            if isinstance(val, (int, float)):
                self.assertFalse(np.isnan(val))
                self.assertFalse(np.isinf(val))

    def test_5_natural_constraints_respected(self):
        """Test 5: Verify non-negativity constraints are respected."""
        res = self.predictor.predict_player("da warner")
        batting = res["batting"]
        self.assertGreaterEqual(batting["expected_runs"], 0.0)
        self.assertGreaterEqual(batting["x_batting_average"], 0.0)
        self.assertGreaterEqual(batting["x_strike_rate"], 0.0)

    def test_6_player_with_batting_history(self):
        """Test 6: Verify player with batting history gets valid batting predictions."""
        res = self.predictor.predict_player("v kohli")
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["has_batting_history"])
        self.assertIn("expected_runs", res["batting"])

    def test_7_player_with_bowling_history(self):
        """Test 7: Verify player with bowling history gets valid bowling predictions."""
        res = self.predictor.predict_player("jj bumrah")
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["has_bowling_history"])
        self.assertIn("x_wickets", res["bowling"])

    def test_8_unknown_player_handling(self):
        """Test 8: Verify unknown player returns clean error status."""
        res = self.predictor.predict_player("non_existent_cricketer_xyz")
        self.assertEqual(res["status"], "error")
        self.assertIn("not found", res["message"])

    def test_9_historical_match_id_leakage_free(self):
        """Test 9: Verify predict_player_for_match(player, match_id) uses data strictly before match_id."""
        player = "da warner"
        df_player = self.predictor.df_pm[self.predictor.df_pm['player'] == player].sort_values('match_id')
        self.assertGreater(len(df_player), 10)
        
        # Pick a historical match_id in the middle
        mid_match = df_player.iloc[5]['match_id']
        res = self.predictor.predict_player_for_match(player, match_id=mid_match)
        
        self.assertEqual(res["status"], "success")
        # Matches used must be exactly 5 (matches strictly before mid_match)
        self.assertEqual(res["historical_matches_used"], 5)
        self.assertLess(res["prediction_point_match_id"], mid_match)

    def test_10_multiple_player_prediction(self):
        """Test 10: Verify batch prediction for multiple players."""
        players = ["v kohli", "da warner", "a kumble"]
        preds = self.predictor.predict_players(players)
        self.assertEqual(len(preds), 3)
        for p_res in preds:
            self.assertEqual(p_res["status"], "success")

def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCricketPredictor)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
