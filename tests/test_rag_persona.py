import tempfile
import unittest
from pathlib import Path

from rag_persona import LocalRAG, PersonaChat, terms


class FakeBackend:
    def __init__(self):
        self.prompt = ""

    def reply(self, prompt, max_new_tokens, temperature):
        self.prompt = prompt
        return "応答です。"


class RagPersonaTests(unittest.TestCase):
    def test_japanese_retrieval(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / "facts.txt").write_text("スファレライトの化学組成はZnSです。", encoding="utf-8")
            rag = LocalRAG(path)
            hits = rag.search("スファレライトの成分は", 2)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].source, "facts.txt")

    def test_zero_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            rag = LocalRAG(Path(d))
            self.assertEqual(rag.search("どこですか"), [])

    def test_dialogue_and_sources(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / "memo.md").write_text("本は静かに読むのが好きです。", encoding="utf-8")
            backend = FakeBackend()
            chat = PersonaChat(LocalRAG(path), backend,
                               {"name": "長門有希", "description": "静か", "style": "簡潔"})
            self.assertEqual(chat.ask("本は好き？"), "応答です。")
            self.assertIn("本は好き？", backend.prompt)
            self.assertEqual(chat.last_sources, ["memo.md"])
            chat.ask("もう少し教えて")
            self.assertIn("応答です。", backend.prompt)

    def test_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / "faq.jsonl").write_text('{"question":"星とは","answer":"恒星です"}\n', encoding="utf-8")
            rag = LocalRAG(path)
            self.assertEqual(len(rag.search("恒星", 1)), 1)

    def test_invalid_chunking(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                LocalRAG(Path(d), chunk_chars=10, overlap=10)

    def test_short_text(self):
        self.assertFalse(terms("a"))


if __name__ == "__main__":
    unittest.main()
