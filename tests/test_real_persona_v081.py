import json
import tempfile
import unittest
from pathlib import Path

from prepare_real_persona_v081 import (
    split_for, dialogue_pairs, compact_prompt, prepare, clean_text,
)


class RealPersonaPreparationV081Tests(unittest.TestCase):
    def sample(self):
        return {"dialogue_id": 1, "interlocutors": ["AA", "AB"], "utterances": [
            {"utterance_id": 0, "interlocutor_id": "AA", "text": "こんにちは。"},
            {"utterance_id": 1, "interlocutor_id": "AB", "text": "こんにちは！"},
            {"utterance_id": 2, "interlocutor_id": "AA", "text": "今日は涼しいですね"},
            {"utterance_id": 3, "interlocutor_id": "AB", "text": "雨が降りましたね"},
        ]}

    def test_parses_sample_structure(self):
        pairs = dialogue_pairs(self.sample(), "04835.json")
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0]["question"], "こんにちは。")
        self.assertEqual(pairs[0]["answer"], "こんにちは！")
        self.assertEqual(pairs[0]["answer_speaker"], "AB")
        self.assertEqual(pairs[0]["conversation_id"], "04835.json")

    def test_context_from_same_reply_speaker(self):
        rows = dialogue_pairs(self.sample(), "a.json", with_context=True)
        self.assertEqual(rows[2]["context"], "こんにちは！")

    def test_duplicate_speaker_not_a_pair(self):
        d = self.sample()
        d["utterances"][1]["interlocutor_id"] = "AA"
        pairs = dialogue_pairs(d, "a.json")
        self.assertEqual(len(pairs), 2)

    def test_long_turn_filtered(self):
        d = self.sample()
        d["utterances"][1]["text"] = "長" * 100
        pairs = dialogue_pairs(d, "a.json", max_answer_chars=30)
        self.assertTrue(all(row["answer"] != "長" * 100 for row in pairs))

    def test_broken_schema(self):
        with self.assertRaises(ValueError):
            dialogue_pairs({"turns": []}, "x.json")

    def test_replacement_character_filtered(self):
        d = self.sample()
        d["utterances"][1]["text"] = "文字�化け"
        pairs = dialogue_pairs(d, "x.json")
        self.assertTrue(all("�" not in row["answer"] for row in pairs))

    def test_splits_stable(self):
        path = Path("04835.json")
        self.assertEqual(split_for(path, 42, .1, .1), split_for(path, 42, .1, .1))
        self.assertIn(split_for(path, 42, .1, .1), ("train", "validation", "test"))

    def test_prompt(self):
        self.assertEqual(compact_prompt({"question": "こんにちは", "context": "", "answer": "はい"}),
                         "問:こんにちは\n答:")

    def test_whitespace_normalization(self):
        self.assertEqual(clean_text(" 今日は\n  涼しい  "), "今日は 涼しい")

    def test_prepare_split_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            dest = root / "out"
            source.mkdir()
            # Enough file-level splits for stable test.
            for i in range(200):
                dialog = self.sample()
                dialog["utterances"][0]["text"] = f"会話{i}。"
                (source / f"{i:05d}.json").write_text(json.dumps(dialog, ensure_ascii=False), encoding="utf-8")
            report = prepare(source, dest)
            self.assertEqual(report["input_files"], 200)
            self.assertGreater(report["records_per_split"]["train"], 0)
            self.assertGreater(report["records_per_split"]["validation"], 0)
            self.assertGreater(report["records_per_split"]["test"], 0)
            ids = []
            for split in ("train", "validation", "test"):
                file = dest / f"real_persona_{split}_v081.jsonl"
                rows = [json.loads(s) for s in file.read_text(encoding="utf-8").splitlines()]
                ids.extend(r["id"] for r in rows)
                self.assertTrue(all(r["conversation_id"] for r in rows))
            self.assertEqual(len(ids), len(set(ids)))

    def test_overwrite_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out = root / "source", root / "out"
            src.mkdir()
            out.mkdir()
            (out / "real_persona_train_v081.jsonl").write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare(src, out)

    def test_no_source_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source"
            src.mkdir()
            with self.assertRaises(ValueError):
                prepare(src, root / "out")


if __name__ == "__main__":
    unittest.main()
