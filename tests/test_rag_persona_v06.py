import json
import tempfile
import unittest
from pathlib import Path

from rag_persona_v03 import LineRAG
from rag_persona_v06 import PlannedPersona, select_excerpt
from prepare_dialogue_v06 import build_examples, split_examples, write_jsonl


class V06Tests(unittest.TestCase):
    def make_chat(self, directory):
        root = Path(directory)
        (root / "knowledge.txt").write_text(
            "宇宙の構造は興味深い。宇宙と時間は観測者に依存する。\n"
            "時空間の時間は観測者によって異なる。\n"
            "Win7のブート時間は短い。\n"
            "宇宙の粒子の動きは興味深い。\n", encoding="utf-8"
        )
        return PlannedPersona(LineRAG(root), {"name": "長門有希"})

    def test_excerpt_is_literal_sentence(self):
        text = "宇宙は広い。時間は観測者に依存する。"
        self.assertEqual(select_excerpt(text, "時間", "宇宙"), "時間は観測者に依存する。")

    def test_excerpt_empty(self):
        self.assertEqual(select_excerpt("", "宇宙", None), "")

    def test_contextual_response(self):
        with tempfile.TemporaryDirectory() as d:
            chat = self.make_chat(d)
            chat.ask("宇宙についてどう考える？")
            response = chat.ask("時間については？")
            self.assertIn("「", response)
            self.assertNotIn("Win7", response)
            self.assertEqual(chat.state.main_topic, "宇宙")
            self.assertEqual(chat.state.focus_topic, "時間")
            self.assertEqual(chat.last_plan.strategy, "EXCERPT")
            self.assertEqual(chat.last_plan.source, "knowledge.txt")
            self.assertIn(chat.last_plan.excerpt, response)

    def test_followup_preserves_topic(self):
        with tempfile.TemporaryDirectory() as d:
            chat = self.make_chat(d)
            chat.ask("宇宙についてどう考える？")
            chat.ask("時間については？")
            chat.ask("もう少し詳しく")
            self.assertEqual(chat.state.focus_topic, "時間")

    def test_identity_is_not_rag(self):
        with tempfile.TemporaryDirectory() as d:
            chat = self.make_chat(d)
            self.assertEqual(chat.ask("あなたの名前は？"), "長門有希。")
            self.assertEqual(chat.last_plan.strategy, "PERSONA")
            self.assertEqual(chat.last_sources, [])

    def test_clear_resets_plan(self):
        with tempfile.TemporaryDirectory() as d:
            chat = self.make_chat(d)
            chat.ask("宇宙とは？")
            chat.clear()
            self.assertEqual(chat.last_plan.strategy, "NONE")
            self.assertIsNone(chat.state.main_topic)

    def test_synthetic_example_preserves_original(self):
        rows = build_examples(["@NagatFakeBot Tweet Messages, (C) 2016", "宇宙は広い。", "日常。"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_line"], 2)
        self.assertIn("「宇宙は広い。」", rows[0]["answer"])
        self.assertEqual(rows[0]["example_type"], "synthetic_extractive_qa")

    def test_split_is_deterministic_disjoint(self):
        rows = [{"id": str(i)} for i in range(15)]
        t1, v1 = split_examples(rows)
        t2, v2 = split_examples(rows)
        self.assertEqual((t1, v1), (t2, v2))
        self.assertFalse({r["id"] for r in t1} & {r["id"] for r in v1})

    def test_write_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "examples.jsonl"
            write_jsonl(path, [{"question": "宇宙", "answer": "宇宙。"}])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8").strip())["question"], "宇宙")
            with self.assertRaises(FileExistsError):
                write_jsonl(path, [])


if __name__ == "__main__":
    unittest.main()
