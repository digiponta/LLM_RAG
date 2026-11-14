"""v0.10.4 source position extraction, swapping and aggregation tests."""
import unittest

from evaluate_evidence_position_v104 import (
    position_question, position_row, swap_pairs, summarize,
)


def case(name, answer, kind="natural"):
    return {
        "id": name, "conversation_id": "line:" + name,
        "question": "コピーしてください", "context": "資料:" + answer,
        "answer": answer, "copy_kind": kind, "example_type": "generalized_copy",
    }


class PositionV104Tests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            case("a", "ABCDEFGHIJ"),
            case("b", "KLMNOPQRST"),
            case("c", "UVWXYZ1234", "synthetic"),
            case("d", "567890abcd", "synthetic"),
        ]

    def test_position_question_first(self):
        self.assertIn("1文字目", position_question(1))

    def test_position_question_last(self):
        self.assertIn("10文字目", position_question(10))

    def test_reject_zero(self):
        with self.assertRaises(ValueError):
            position_question(0)

    def test_reject_boolean(self):
        with self.assertRaises(ValueError):
            position_question(True)

    def test_first_char(self):
        self.assertEqual(position_row(self.rows[0], 1)["answer"], "A")

    def test_fifth_char(self):
        self.assertEqual(position_row(self.rows[0], 5)["answer"], "E")

    def test_last_char(self):
        self.assertEqual(position_row(self.rows[0], 10)["answer"], "J")

    def test_reject_past_end(self):
        with self.assertRaises(ValueError):
            position_row(self.rows[0], 11)

    def test_reject_wrong_type(self):
        with self.assertRaises(ValueError):
            position_row({**self.rows[0], "example_type": "other"}, 1)

    def test_swap_uses_different_evidence(self):
        pairs = swap_pairs(self.rows)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all(a["answer"] != b["answer"] for a, b in pairs))

    def test_swap_preserves_kind(self):
        pairs = swap_pairs(self.rows)
        self.assertTrue(all(a["copy_kind"] == b["copy_kind"] for a, b in pairs))

    def test_swap_deterministic(self):
        self.assertEqual(swap_pairs(self.rows), swap_pairs(self.rows))

    def test_swap_single_member_kind_rejected(self):
        with self.assertRaises(ValueError):
            swap_pairs([self.rows[0], self.rows[1], self.rows[2]])

    def test_swap_rejects_duplicate_answers(self):
        with self.assertRaises(ValueError):
            swap_pairs([self.rows[0], dict(self.rows[0])])

    def test_swapped_gold_follows_donor(self):
        donor = self.rows[1]
        swapped = position_row(self.rows[0], 5, evidence=donor["answer"])
        self.assertEqual(swapped["answer"], "O")
        self.assertEqual(swapped["question"], position_row(self.rows[0], 5)["question"])

    def test_summary_separates_variants(self):
        a = {"position": 1, "kind": "natural", "variant": "original",
             "baseline": {"exact_character": True, "first_token_correct": True,
                          "first_token_rank": 1, "first_token_probability": .9,
                          "gold_token_count": 1}}
        b = {"position": 1, "kind": "natural", "variant": "swapped",
             "baseline": {"exact_character": False, "first_token_correct": False,
                          "first_token_rank": 10, "first_token_probability": .01,
                          "gold_token_count": 2}}
        result = summarize([a, b], "baseline")
        self.assertEqual(result["original"]["exact_character_fraction"], 1)
        self.assertEqual(result["swapped"]["exact_character_fraction"], 0)
        self.assertEqual(result["position_1"]["cases"], 2)
        self.assertEqual(result["position_1"]["single_token_target_fraction"], .5)


if __name__ == "__main__":
    unittest.main()
