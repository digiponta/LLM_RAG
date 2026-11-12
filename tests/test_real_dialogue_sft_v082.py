"""CPU-only regression tests for v0.8.2 Real Persona Chat SFT."""
import json
import tempfile
import unittest
from pathlib import Path

from train_real_dialogue_v082 import (
    load_dialogue_rows, encode_dialogue, prepare_sequences,
)


class FakeTokenizer:
    unk_id = 1
    pad_id = 0
    eos_id = 3

    def encode(self, text, add_bos=False):
        seq = [1 if c == "★" else ord(c) + 10 for c in text]
        return ([2] if add_bos else []) + seq


class RealDialogueSFTV082Tests(unittest.TestCase):
    def sample(self, n=15):
        return [{"id": f"c{i}:1", "conversation_id": f"c{i}",
                 "question": f"元気ですか{i}", "answer": "はい、元気です。"} for i in range(n)]

    def test_encode_question_and_answer(self):
        row = self.sample(1)[0]
        x, y = encode_dialogue(row, FakeTokenizer(), 64)[0]
        self.assertEqual(len(x), len(y))
        self.assertEqual(y[-1], 3)
        self.assertEqual(x[0], 2)
        self.assertIn(-100, y)

    def test_answer_start_unmasked(self):
        row = self.sample(1)[0]
        tokenizer = FakeTokenizer()
        prompt = tokenizer.encode("問:" + row["question"] + "\n答:", add_bos=True)
        seq, reason = encode_dialogue(row, tokenizer, 64)
        self.assertEqual(reason, "accepted")
        self.assertEqual(seq[1][len(prompt) - 1], tokenizer.encode(row["answer"])[0])
        self.assertTrue(all(y == -100 for y in seq[1][:len(prompt)-1]))

    def test_long_answer_not_truncated(self):
        row = self.sample(1)[0]
        row["answer"] = "会" * 80
        seq, reason = encode_dialogue(row, FakeTokenizer(), 64)
        self.assertIsNone(seq)
        self.assertEqual(reason, "over_context")

    def test_long_question_not_truncated(self):
        row = self.sample(1)[0]
        row["question"] = "問" * 60
        seq, reason = encode_dialogue(row, FakeTokenizer(), 64)
        self.assertIsNone(seq)
        self.assertEqual(reason, "over_context")

    def test_unk_reject(self):
        row = self.sample(1)[0]
        row["answer"] = "★★★"
        seq, reason = encode_dialogue(row, FakeTokenizer(), 64, max_unknown_fraction=0.01)
        self.assertEqual(reason, "high_unk")
        self.assertIsNone(seq)

    def test_unk_allowed(self):
        row = self.sample(1)[0]
        row["answer"] = "★★★"
        seq, reason = encode_dialogue(row, FakeTokenizer(), 64, max_unknown_fraction=1)
        self.assertEqual(reason, "accepted")
        self.assertIsNotNone(seq)

    def test_short_answer_reject(self):
        row = self.sample(1)[0]
        row["answer"] = "う"
        _, reason = encode_dialogue(row, FakeTokenizer(), 64)
        self.assertEqual(reason, "too_short")

    def test_prepare_stats(self):
        rows = self.sample(3)
        pairs, stats = prepare_sequences(rows, FakeTokenizer(), 64, .1)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(stats.get("accepted"), 3)

    def test_load_seeded_sample_stable(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "train.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.sample()), encoding="utf-8")
            a = load_dialogue_rows(path, 5, 42)
            b = load_dialogue_rows(path, 5, 42)
            self.assertEqual([r["id"] for r in a], [r["id"] for r in b])
            self.assertEqual(len(a), 5)

    def test_limit_zero_means_all(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "train.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.sample()), encoding="utf-8")
            self.assertEqual(len(load_dialogue_rows(path, 0, 42)), 15)

    def test_bad_row_rejected(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "train.jsonl"
            path.write_text('{"id":"1","question":"hello"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dialogue_rows(path, 5, 42)

    def test_duplicate_id_rejected(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "train.jsonl"
            record = json.dumps(self.sample(1)[0], ensure_ascii=False)
            path.write_text(record + "\n" + record + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dialogue_rows(path, 5, 42)


if __name__ == "__main__":
    unittest.main()
