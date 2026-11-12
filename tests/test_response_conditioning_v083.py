"""Tests for v0.8.3 response bias and question-conditioning data utilities."""
import unittest

from rebalance_responses_v083 import (
    normalize_answer, response_bias, balanced_sample,
)
from evaluate_conditioning_v083 import swapped_question_rows


class ResponseConditioningV083Tests(unittest.TestCase):
    def rows(self):
        return [
            {"id": f"c{i}", "question": f"質問{i}", "answer": "そうですね。" if i < 6 else f"回答{i}。"}
            for i in range(12)
        ]

    def test_normalize_answer(self):
        self.assertEqual(normalize_answer(" そうですね。 "), "そうですね")

    def test_frequency_count(self):
        metric = response_bias(self.rows())
        self.assertEqual(metric["sample_count"], 12)
        self.assertEqual(metric["most_frequent_answer_share"], .5)

    def test_default_no_raw_text(self):
        metric = response_bias(self.rows())
        self.assertNotIn("answer", metric["top_answers"][0])
        self.assertIn("answer_hash", metric["top_answers"][0])

    def test_explicit_text(self):
        metric = response_bias(self.rows(), expose_text=True)
        self.assertEqual(metric["top_answers"][0]["answer"], "そうですね")

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            response_bias([])

    def test_balance_caps_generic(self):
        selected, report = balanced_sample(self.rows(), 10, 2, 42)
        self.assertLessEqual(sum(normalize_answer(r["answer"]) == "そうですね" for r in selected), 2)
        self.assertLess(report["selected"], 10)

    def test_balanced_deterministic(self):
        a, _ = balanced_sample(self.rows(), 8, 2, 42)
        b, _ = balanced_sample(self.rows(), 8, 2, 42)
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            balanced_sample(self.rows(), 0, 1, 42)
        with self.assertRaises(ValueError):
            balanced_sample(self.rows(), 1, 0, 42)

    def test_swapped_question_not_same(self):
        rows = self.rows()
        mismatches = swapped_question_rows(rows, 42)
        self.assertEqual(len(mismatches), len(rows))
        self.assertTrue(all(rows[i]["question"] != r["question"] for i, r in mismatches))
        self.assertTrue(all(rows[i]["answer"] == r["answer"] for i, r in mismatches))

    def test_swapped_deterministic(self):
        self.assertEqual(swapped_question_rows(self.rows(), 42),
                         swapped_question_rows(self.rows(), 42))

    def test_swapped_clears_context(self):
        rows = self.rows()
        for row in rows:
            row["context"] = "前の発話"
        self.assertTrue(all(r["context"] == "" for _, r in swapped_question_rows(rows, 42)))

    def test_swapped_needs_two(self):
        with self.assertRaises(ValueError):
            swapped_question_rows(self.rows()[:1], 42)


if __name__ == "__main__":
    unittest.main()
