"""CPU-only tests for controlled position-ratio v0.10.9 data and EOS diagnostic."""
import unittest
from collections import Counter

from prepare_position_ratio_v109 import WEIGHT_SWEEP, balanced_positions
from prepare_balanced_multitask_v107 import quota, TASKS
from evaluate_position_learning_v105 import aggregate


def sample(pos, n):
    return {"id": f"id:{pos}:{n}", "char_position": pos}


class RatioSweepTests(unittest.TestCase):
    def setUp(self):
        self.rows = [sample(p, n) for p in (1, 5, 10) for n in range(50)]

    def test_four_weights_sum(self):
        self.assertEqual(sorted(WEIGHT_SWEEP), [5, 10, 20, 30])
        for pct, weights in WEIGHT_SWEEP.items():
            self.assertEqual(sum(weights.values()), 100)
            self.assertEqual(weights["position"], pct)
            self.assertEqual(weights["copy"], 5)

    def test_quotas_10(self):
        self.assertEqual(quota(4000, WEIGHT_SWEEP[10])["position"], 400)

    def test_quotas_20(self):
        self.assertEqual(quota(4000, WEIGHT_SWEEP[20])["position"], 800)

    def test_quotas_30(self):
        self.assertEqual(quota(4000, WEIGHT_SWEEP[30])["position"], 1200)

    def test_position_stratification(self):
        rows = balanced_positions(self.rows, 90)
        self.assertEqual(Counter(r["char_position"] for r in rows),
                         Counter({1: 30, 5: 30, 10: 30}))

    def test_position_remainder(self):
        rows = balanced_positions(self.rows, 11)
        self.assertEqual(Counter(r["char_position"] for r in rows),
                         Counter({1: 4, 5: 4, 10: 3}))

    def test_position_deterministic(self):
        self.assertEqual(balanced_positions(self.rows, 21, 22),
                         balanced_positions(self.rows, 21, 22))

    def test_position_no_duplicate_ids(self):
        rows = balanced_positions(self.rows, 90)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))

    def test_position_overflow_rejected(self):
        with self.assertRaises(ValueError):
            balanced_positions(self.rows, 200)

    def test_position_too_small_rejected(self):
        with self.assertRaises(ValueError):
            balanced_positions(self.rows, 2)

    def test_bad_position_rejected(self):
        with self.assertRaises(ValueError):
            balanced_positions(self.rows + [sample(7, 0)], 15)

    def test_empty_response_aggregate(self):
        row = {
            "kind": "synthetic", "position": 1, "variant": "original",
            "tuned": {
                "exact": False, "first_token_correct": False,
                "gold_eos_probability": .99, "first_step_eos_probability": .98,
                "empty_response": True, "ended_by_eos": True,
                "generated_chars": 0,
            },
        }
        scored = aggregate([row], "tuned")["all"]
        self.assertEqual(scored["empty_response_fraction"], 1)
        self.assertAlmostEqual(scored["mean_first_step_eos_probability"], .98)
        self.assertAlmostEqual(scored["mean_gold_eos_probability"], .99)

    def test_eos_can_be_correct_but_output_empty(self):
        row = {
            "kind": "natural", "position": 10, "variant": "original",
            "tuned": {
                "exact": False, "first_token_correct": False,
                "gold_eos_probability": .1, "first_step_eos_probability": .95,
                "empty_response": True, "ended_by_eos": True,
                "generated_chars": 0,
            },
        }
        scored = aggregate([row], "tuned")["all"]
        self.assertEqual(scored["eos_termination_fraction"], 1)
        self.assertEqual(scored["exact_rate"], 0)


if __name__ == "__main__":
    unittest.main()
