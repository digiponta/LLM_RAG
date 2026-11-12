import tempfile
import unittest
from pathlib import Path

from rag_persona_v03 import ExtractivePersona, LineRAG, extract_topic


class V03Tests(unittest.TestCase):
    def make_chat(self, root):
        return ExtractivePersona(LineRAG(root), {"name": "長門有希"})

    def test_topic_extraction(self):
        self.assertEqual(extract_topic("宇宙とは何ですか？"), "宇宙")
        self.assertEqual(extract_topic("宇宙についてどう考える？"), "宇宙")

    def test_persona(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = self.make_chat(Path(tmp))
            self.assertEqual(chat.ask("あなたの名前は？"), "長門有希。")
            self.assertEqual(chat.last_sources, [])

    def test_exact_quote_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nagato.txt").write_text("宇宙は全体として膨張している。\n読書について考える。\n", encoding="utf-8")
            chat = self.make_chat(root)
            self.assertIn("「宇宙は全体として膨張している。」", chat.ask("宇宙とは何ですか？"))
            self.assertEqual(chat.last_sources, ["nagato.txt"])

    def test_no_unrelated_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nagato.txt").write_text("宇宙の概念。\n", encoding="utf-8")
            chat = self.make_chat(root)
            self.assertIn("見つけられなかった", chat.ask("スファレライトとは？"))
            self.assertEqual(chat.last_sources, [])

    def test_one_line_per_passage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nagato.txt").write_text("@NagatFakeBot Tweet Messages, (C) 2016\n宇宙。\n量子力学。\n", encoding="utf-8")
            self.assertEqual([p.text for p in LineRAG(root).passages], ["宇宙。", "量子力学。"])

    def test_llm_requires_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                ExtractivePersona(LineRAG(Path(tmp)), {"name": "長門有希"}, mode="llm")


if __name__ == "__main__":
    unittest.main()
