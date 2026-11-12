import json
import tempfile
import unittest
from pathlib import Path

from prepare_context_dialogue_v087 import (
    previous_turns, context_for, build_paired, output_files,
)


class FakeTokenizer:
    unk_id = 999
    eos_id = 2
    def encode(self, text, add_bos=False):
        return ([1] if add_bos else []) + [ord(c) for c in text]


def fixture():
    return {"utterances": [
        {"interlocutor_id": "AA", "text": "冬は寒いですね"},
        {"interlocutor_id": "AB", "text": "厚着しています"},
        {"interlocutor_id": "AA", "text": "あなたは？"},
        {"interlocutor_id": "AB", "text": "もこもこ着ています"},
    ]}


class ContextDatasetV087Tests(unittest.TestCase):
    def test_previous_turns(self):
        self.assertEqual(previous_turns(fixture(), 3, 2),
                         ["冬は寒いですね", "厚着しています"])

    def test_one_turn(self):
        self.assertEqual(previous_turns(fixture(), 3, 1), ["厚着しています"])

    def test_context_join(self):
        self.assertEqual(context_for(fixture(), 3, 2), "冬は寒いですね／厚着しています")

    def test_no_prior_turn(self):
        self.assertEqual(previous_turns(fixture(), 1, 2), [])

    def test_no_cross_speaker_splice(self):
        data = fixture()
        data["utterances"][1]["interlocutor_id"] = "AA"
        self.assertEqual(previous_turns(data, 3, 2), [])

    def test_malformed_history(self):
        data = fixture()
        data["utterances"][1]["text"] = "文字�化け"
        self.assertEqual(previous_turns(data, 3, 2), [])

    def test_bad_answer_index(self):
        self.assertEqual(previous_turns(fixture(), 99, 2), [])

    def test_matched_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text(json.dumps(fixture(), ensure_ascii=False), encoding="utf-8")
            row = {"id": "a.json:3", "conversation_id": "a.json",
                   "question": "あなたは？", "answer": "もこもこ着ています"}
            plain, context, counts = build_paired([row], root, FakeTokenizer(), 64, 2, .1)
            self.assertEqual(counts["paired_accepted"], 1)
            self.assertEqual(plain[0]["id"], context[0]["id"])
            self.assertEqual(plain[0]["context"], "")
            self.assertEqual(context[0]["context"], "冬は寒いですね／厚着しています")
            self.assertEqual(context[0]["answer"], plain[0]["answer"])

    def test_small_context_is_counted_not_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text(json.dumps(fixture(), ensure_ascii=False), encoding="utf-8")
            row = {"id": "a.json:3", "conversation_id": "a.json",
                   "question": "あなたは？", "answer": "もこもこ着ています"}
            plain, context, counts = build_paired([row], root, FakeTokenizer(), 25, 2, .1)
            self.assertEqual(len(plain), 0)
            self.assertEqual(len(context), 0)
            self.assertGreater(counts.get("context_over_context", 0), 0)

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text(json.dumps(fixture(), ensure_ascii=False), encoding="utf-8")
            row = {"id": "a.json:3", "conversation_id": "a.json",
                   "question": "あなたは？", "answer": "異なる答え"}
            plain, context, counts = build_paired([row], root, FakeTokenizer(), 64, 2, .1)
            self.assertEqual(len(plain), 0)
            self.assertEqual(counts["source_row_changed"], 1)

    def test_directory_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = {"id": "../outside.json:3", "conversation_id": "../outside.json",
                   "question": "あなたは？", "answer": "回答"}
            with self.assertRaises(ValueError):
                build_paired([row], Path(tmp), FakeTokenizer(), 64, 2, .1)

    def test_output_files_distinct(self):
        files = output_files(Path("out"))
        self.assertEqual(len(files), 6)
        self.assertEqual(len(set(files.values())), 6)


if __name__ == "__main__":
    unittest.main()
