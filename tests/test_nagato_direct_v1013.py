"""Local Nagato proper-name one-sentence shortcut regression tests (no GPU)."""
import tempfile
import unittest
from pathlib import Path

from nagato_direct_v1013 import (
    NagatoDirect, candidate_names, sentences_from_line, normalized)


class NagatoDirectTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.path = Path(self.workspace.name) / "data-nagato.txt"
        self.path.write_text(
            "@NagatFakeBot header\n"
            "イスカンダルには伝説がある。次の文章は別の文です。\n"
            "攻殻機動隊の世界を考える。読書は好きです。\n"
            "宇宙の果てには境界がない。月について考える。\n"
            "ある資料に長門有希が登場した。長門有希について二つ目の文章。\n",
            encoding="utf-8")
        self.direct = NagatoDirect(self.path)

    def test_returns_only_one_sentence(self):
        hit = self.direct.find("イスカンダルについて教えて")
        self.assertEqual(hit.sentence, "イスカンダルには伝説がある。")
        self.assertEqual(hit.line_number, 2)

    def test_kanji_proper_name(self):
        hit = self.direct.find("攻殻機動隊とは？")
        self.assertEqual(hit.sentence, "攻殻機動隊の世界を考える。")

    def test_name_not_in_question_returns_none(self):
        self.assertIsNone(self.direct.find("こんにちは"))

    def test_common_word_does_not_trigger(self):
        self.assertIsNone(self.direct.find("宇宙について教えて"))

    def test_query_requires_literal_name(self):
        self.assertIsNone(self.direct.find("攻殻機動は何ですか"))

    def test_source_not_quoted_header(self):
        self.assertIsNone(self.direct.find("@NagatFakeBot"))

    def test_sources_original_first_match(self):
        hit = self.direct.find("長門有希とは")
        self.assertEqual(hit.sentence, "ある資料に長門有希が登場した。")

    def test_offline_missing_file(self):
        absent = NagatoDirect(self.path.with_name("missing.txt"))
        self.assertFalse(absent.available)
        self.assertIsNone(absent.find("長門有希"))

    def test_reload(self):
        self.path.write_text("コスモクリーナーは架空の装置。", encoding="utf-8")
        self.direct.refresh()
        self.assertIsNone(self.direct.find("イスカンダル"))
        self.assertEqual(self.direct.find("コスモクリーナー").sentence,
                         "コスモクリーナーは架空の装置。")

    def test_user_entity_file_kanji(self):
        entity_file = self.path.with_name("entities.txt")
        entity_file.write_text("# 任意の固有名詞\n固有作品名\n", encoding="utf-8")
        self.path.write_text("固有作品名が登場する。続きの文です。", encoding="utf-8")
        index = NagatoDirect(self.path, entity_file=entity_file)
        self.assertEqual(index.find("固有作品名とは").sentence, "固有作品名が登場する。")

    def test_invalid_name_not_found_in_source(self):
        entity_file = self.path.with_name("entities.txt")
        entity_file.write_text("存在しない固有名詞\n", encoding="utf-8")
        index = NagatoDirect(self.path, entity_file=entity_file)
        self.assertIsNone(index.find("存在しない固有名詞"))

    def test_split_keeps_punctuation(self):
        self.assertEqual(
            sentences_from_line("最初の文。次の文！最後は終止符なし"),
            ["最初の文。", "次の文！", "最後は終止符なし"])

    def test_skip_header(self):
        self.assertEqual(sentences_from_line("@NagatFakeBot copyright"), [])

    def test_entity_lexicon_recognizes_source_names(self):
        names = candidate_names("コスモクリーナーがある。イスカンダル。")
        self.assertIn("コスモクリーナー", names)

    def test_normalized_whitespace_case(self):
        self.assertEqual(normalized("  PYTHON \n"), "python")

    def test_no_modified_original_sentence(self):
        self.path.write_text("イスカンダル は星。 これは第二文。\n", encoding="utf-8")
        self.direct.refresh()
        self.assertEqual(self.direct.find("イスカンダルとは").sentence,
                         "イスカンダル は星。")


if __name__ == "__main__":
    unittest.main()
