import json
import tempfile
import unittest
from pathlib import Path

from train_dialogue_v07 import (
    read_jsonl, short_answer, prompt_for, make_supervised_sequence, prepare_rows,
)
from compare_dialogue_v07 import has_japanese


class FakeTokenizer:
    unk_id = 1
    eos_id = 3
    def encode(self, text, add_bos=False):
        values = [ord(c) + 10 for c in text]
        return ([2] if add_bos else []) + values


class DialogueV07Tests(unittest.TestCase):
    def test_short_answer_verbatim(self):
        row = {"topic": "宇宙", "answer": "資料では、こう記されている。\n「時間。宇宙は広い。さらに。」"}
        self.assertEqual(short_answer(row), "宇宙は広い。")

    def test_prompt_short(self):
        self.assertEqual(prompt_for("宇宙"), "問:宇宙\n答:")

    def test_masks_question_only(self):
        tok = FakeTokenizer()
        prefix = tok.encode("問:宇宙\n答:", add_bos=True)
        pair = make_supervised_sequence(tok, "宇宙", "宇宙は広い宇宙だ。", 64)
        self.assertIsNotNone(pair)
        x, y = pair
        self.assertEqual(len(x), len(y))
        self.assertEqual(y[:len(prefix)-1], [-100]*(len(prefix)-1))
        self.assertEqual(y[len(prefix)-1], ord("宇")+10)
        self.assertEqual(y[-1], tok.eos_id)

    def test_no_prompt_truncation(self):
        self.assertIsNone(make_supervised_sequence(FakeTokenizer(), "とても長い質問です"*7,
                                                    "宇宙とは大変興味深いものです。", 64))

    def test_short_answer_filtered(self):
        self.assertIsNone(make_supervised_sequence(FakeTokenizer(), "宇宙", "短い", 64))

    def test_read_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jsonl"
            p.write_text(json.dumps({"id": "1", "topic": "宇宙", "answer": "「宇宙は広い。」"},
                                    ensure_ascii=False)+"\n", encoding="utf-8")
            self.assertEqual(len(read_jsonl(p)), 1)

    def test_invalid_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.jsonl"
            p.write_text('{"topic":"宇宙"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_jsonl(p)

    def test_unk_threshold(self):
        tok = FakeTokenizer()
        row = [{"id": "a", "topic": "宇宙", "answer": "「宇宙の動向は考察に値する。」"}]
        rows, skipped, _, _ = prepare_rows(row, tok, 64, 0.03)
        self.assertEqual((len(rows), skipped), (1, 0))

    def test_unicode_quality_flag(self):
        self.assertTrue(has_japanese("宇宙"))
        self.assertFalse(has_japanese("123 xyz"))

if __name__ == "__main__":
    unittest.main()
