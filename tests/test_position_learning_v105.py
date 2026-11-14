"""Position learning v0.10.5: grouping, split safety, and metrics."""
import unittest

from prepare_position_learning_v105 import transform, guard
from evaluate_position_learning_v105 import pair_metrics, aggregate


class Tokenizer:
    unk_id = 3
    eos_id = 2

    def encode(self, text, add_bos=False):
        return ([1] if add_bos else []) + [
            3 if ch == "★" else 10 + (ord(ch) % 128) for ch in text
        ]


def source(name, kind="natural", text="ABCDEFGHIJ"):
    return {
        "id": name, "conversation_id": "source:" + name,
        "question": "copy", "answer": text, "context": "資料:" + text,
        "example_type": "generalized_copy", "copy_kind": kind,
    }


def metric(exact, first=True, eos=.9, stopped=True):
    return {"exact": exact, "first_token_correct": first,
            "gold_eos_probability": eos, "ended_by_eos": stopped,
            "generated_chars": 1, "generated_hash": "match" if exact else "wrong"}


class V105Tests(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()

    def test_three_positions_per_source(self):
        rows, counts = transform([source("a")], self.tok, split="train")
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["char_position"] for r in rows}, {1, 5, 10})
        self.assertEqual(counts["position_5_accepted"], 1)

    def test_target_indices(self):
        rows, _ = transform([source("a")], self.tok, split="train")
        self.assertEqual([r["answer"] for r in rows], ["A", "E", "J"])

    def test_shared_source_group_id(self):
        rows, _ = transform([source("a")], self.tok, split="train")
        self.assertEqual(len({r["conversation_id"] for r in rows}), 1)

    def test_positions_deterministic(self):
        a, _ = transform([source("a")], self.tok, split="train")
        b, _ = transform([source("a")], self.tok, split="train")
        self.assertEqual(a, b)

    def test_reject_position_zero(self):
        with self.assertRaises(ValueError):
            transform([source("a")], self.tok, split="train", positions=[0])

    def test_reject_duplicate_positions(self):
        with self.assertRaises(ValueError):
            transform([source("a")], self.tok, split="train", positions=[1, 1])

    def test_reject_too_short_source(self):
        with self.assertRaises(ValueError):
            transform([source("a", text="AB")], self.tok, split="train")

    def test_reject_unknown_gold_source_group(self):
        rows, stats = transform([source("a", text="ABCD★FGHIJ"), source("b")],
                                self.tok, split="train")
        self.assertEqual(len(rows), 3)
        self.assertEqual(stats["source_rejected_unknown_target"], 1)

    def test_guard_safe(self):
        train, _ = transform([source("a")], self.tok, split="train")
        dev, _ = transform([source("b", text="KLMNOPQRST")],
                           self.tok, split="development")
        guard(train, dev)

    def test_guard_reject_source_overlap(self):
        train, _ = transform([source("a")], self.tok, split="train")
        dev, _ = transform([source("a")], self.tok, split="train")
        with self.assertRaises(ValueError):
            guard(train, dev)

    def test_guard_reject_same_text(self):
        train, _ = transform([source("a")], self.tok, split="train")
        dev, _ = transform([source("b")], self.tok, split="development")
        with self.assertRaises(ValueError):
            guard(train, dev)

    def test_pair_change_vs_correct_distinct(self):
        old = {"gold": "A", "baseline": {**metric(False), "generated_hash": "X"}}
        new = {"gold": "B", "baseline": {**metric(False), "generated_hash": "Y"}}
        scores = pair_metrics([(old, new)], "baseline")
        self.assertEqual(scores["output_change_fraction"], 1)
        self.assertEqual(scores["changed_gold_and_both_correct_fraction"], 0)

    def test_pair_both_correct(self):
        old = {"gold": "A", "tuned": metric(True)}
        new = {"gold": "B", "tuned": metric(True)}
        scores = pair_metrics([(old, new)], "tuned")
        self.assertEqual(scores["changed_gold_and_both_correct_fraction"], 1)

    def test_pair_same_gold_excluded_from_conditional(self):
        old = {"gold": "A", "tuned": metric(True)}
        new = {"gold": "A", "tuned": metric(True)}
        scores = pair_metrics([(old, new)], "tuned")
        self.assertEqual(scores["gold_change_fraction"], 0)
        self.assertIsNone(scores["changed_gold_and_both_correct_given_changed_gold"])

    def test_empty_pairs_rejected(self):
        with self.assertRaises(ValueError):
            pair_metrics([], "tuned")

    def test_eos_aggregation(self):
        entry = {"position": 5, "kind": "natural", "variant": "original",
                 "baseline": metric(True, eos=.25, stopped=False)}
        summary = aggregate([entry], "baseline")
        self.assertEqual(summary["position_5"]["mean_gold_eos_probability"], .25)
        self.assertEqual(summary["position_5"]["eos_termination_fraction"], 0)

    def test_one_char_answer_unmasked(self):
        from train_real_dialogue_v082 import encode_dialogue, prepare_sequences
        row = {"id": "1", "conversation_id": "1", "question": "何文字目",
               "context": "資料:ABCDEFGHIJ", "answer": "A"}
        encoded, reason = encode_dialogue(row, self.tok, 256, min_answer_tokens=1)
        self.assertEqual(reason, "accepted")
        self.assertEqual(encoded[1][-1], self.tok.eos_id)
        default_pairs, _ = prepare_sequences([row], self.tok, 256, .1)
        enabled_pairs, _ = prepare_sequences([row], self.tok, 256, .1, min_answer_tokens=1)
        self.assertEqual(len(default_pairs), 0)
        self.assertEqual(len(enabled_pairs), 1)


if __name__ == "__main__":
    unittest.main()
