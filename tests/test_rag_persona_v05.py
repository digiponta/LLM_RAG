import tempfile
import unittest
from pathlib import Path
from rag_persona_v03 import LineRAG
from rag_persona_v05 import SemanticPersona, semantic_rank, related_followon

CORPUS = (
    "宇宙について観測者は異なる視点を持つ。\n"
    "宇宙の粒子と時間について考える。観測者が重要だ。\n"
    "時空間における時間のあり方は観測者の立場による。\n"
    "宇宙の時間は重力と相対性に関係する。\n"
    "Win7のブート時間を30秒短縮した。\n"
    "CPUの時間を節約できる。\n"
    "読書は静かな場所で行う。\n"
)

class SemanticRAGV05Tests(unittest.TestCase):
    def make_chat(self, root):
        (root / "knowledge.txt").write_text(CORPUS, encoding="utf-8")
        return SemanticPersona(LineRAG(root), {"name": "長門有希"})

    def test_related_followon(self):
        self.assertTrue(related_followon("時間については？", "宇宙"))
        self.assertFalse(related_followon("時間については？", None))
        self.assertFalse(related_followon("時間とは？", "宇宙"))

    def test_contextual_time_not_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = self.make_chat(Path(tmp))
            chat.ask("宇宙についてどう考える？")
            reply = chat.ask("時間については？")
            self.assertNotIn("Win7", reply)
            self.assertNotIn("CPU", reply)
            self.assertIn("時間", reply)
            self.assertEqual(chat.state.main_topic, "宇宙")
            self.assertEqual(chat.state.focus_topic, "時間")

    def test_followup_is_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = self.make_chat(Path(tmp))
            chat.ask("宇宙についてどう考える？")
            first = chat.ask("時間については？")
            second = chat.ask("もう少し詳しく")
            self.assertNotEqual(first, second)
            self.assertEqual(chat.state.focus_topic, "時間")

    def test_new_root_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = self.make_chat(Path(tmp))
            chat.ask("宇宙についてどう考える？")
            chat.ask("読書とは？")
            self.assertEqual(chat.state.main_topic, "読書")

    def test_clear_forgets_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = self.make_chat(Path(tmp))
            chat.ask("宇宙とは？")
            chat.clear()
            self.assertIsNone(chat.state.main_topic)
            self.assertIn("どの話題", chat.ask("もっと詳しく"))

    def test_topic_mismatch_refuses_quote(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = self.make_chat(Path(tmp))
            chat.ask("宇宙とは？")
            answer = chat.ask("スファレライトについては？")
            self.assertIn("見つからなかった", answer)
            self.assertEqual(chat.last_sources, [])

    def test_seen_passage_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = self.make_chat(Path(tmp))
            seen = {"宇宙の粒子と時間について考える。観測者が重要だ。"}
            matches = semantic_rank(chat.rag.passages, "時間", "宇宙", "TOPIC", seen)
            self.assertTrue(matches)
            self.assertTrue(all(item.text not in seen for item in matches))

    def test_identity_preserves_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = self.make_chat(Path(tmp))
            chat.ask("宇宙とは？")
            self.assertEqual(chat.ask("あなたの名前は？"), "長門有希。")
            self.assertEqual(chat.state.main_topic, "宇宙")

    def test_no_context_followup(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = self.make_chat(Path(tmp))
            self.assertIn("どの話題", chat.ask("もう少し詳しく"))

if __name__ == "__main__":
    unittest.main()
