import tempfile
import unittest
from pathlib import Path

from rag_persona_v03 import LineRAG
from rag_persona_v04 import ContextPersona, intent_of, explicit_topic, rank_passages


class ContextPersonaV04Tests(unittest.TestCase):
    def setup_chat(self, root):
        (root / "nagato.txt").write_text(
            "宇宙は全体として膨張している。\n"
            "宇宙という概念は興味深い。\n"
            "時間は観測者に依存する可能性がある。\n"
            "時間という概念を考える。\n"
            "量子力学には可能性がある。\n", encoding="utf-8")
        return ContextPersona(LineRAG(root), {"name": "長門有希"})

    def test_intent(self):
        self.assertEqual(intent_of("宇宙とは何ですか？"), "DEFINITION")
        self.assertEqual(intent_of("宇宙についてどう思う？"), "OPINION")
        self.assertEqual(intent_of("もう少し詳しく"), "FOLLOWUP")

    def test_topic_extraction(self):
        self.assertEqual(explicit_topic("宇宙とは何ですか？", "DEFINITION"), "宇宙")
        self.assertEqual(explicit_topic("時間については？", "TOPIC"), "時間")
        self.assertIsNone(explicit_topic("もっと詳しく", "FOLLOWUP"))

    def test_followup_retains_topic_and_avoids_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = self.setup_chat(Path(directory))
            first = chat.ask("宇宙とは何ですか？")
            second = chat.ask("もう少し詳しく")
            self.assertNotEqual(first, second)
            self.assertEqual(chat.state.topic, "宇宙")
            self.assertEqual(len(chat.state.seen), 2)

    def test_explicit_new_topic_resets_novelty(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = self.setup_chat(Path(directory))
            chat.ask("宇宙についてどう思う？")
            reply = chat.ask("時間については？")
            self.assertEqual(chat.state.topic, "時間")
            self.assertIn("時間", reply)
            self.assertEqual(len(chat.state.seen), 1)

    def test_clear_resets_context(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = self.setup_chat(Path(directory))
            chat.ask("宇宙とは？")
            chat.clear()
            self.assertIsNone(chat.state.topic)
            self.assertIn("どの話題", chat.ask("もっと詳しく"))

    def test_no_unrelated_quote(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = self.setup_chat(Path(directory))
            reply = chat.ask("スファレライトとは何ですか？")
            self.assertIn("見つからなかった", reply)
            self.assertEqual(chat.last_sources, [])

    def test_identity_does_not_destroy_topic(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = self.setup_chat(Path(directory))
            chat.ask("宇宙とは？")
            self.assertEqual(chat.ask("あなたの名前は？"), "長門有希。")
            self.assertEqual(chat.state.topic, "宇宙")

    def test_rank_novelty(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = self.setup_chat(Path(directory))
            used = {"宇宙は全体として膨張している。"}
            matches = rank_passages(chat.rag, "宇宙", "TOPIC", used)
            self.assertTrue(matches)
            self.assertNotIn(matches[0].text, used)


if __name__ == "__main__":
    unittest.main()
