"""v0.10.12 CLI: frozen unified prompt parity and guarded local commands."""
import unittest

from chat import (
    CHAT_MODES, POSITION_LIMIT, copy_prompt, parse_position_command,
    position_prompt, make_prompt)
from evaluate_evidence_position_v104 import position_question
from prepare_generalized_copy_v103 import QUESTION as COPY_QUESTION
from prepare_real_persona_v081 import compact_prompt


class FakeTokenizer:
    def encode(self, text, add_bos=False):
        return ([2] if add_bos else []) + list(range(len(text)))


class ChatV1012Tests(unittest.TestCase):
    def test_trained_position_1(self):
        question = position_question(1)
        self.assertEqual(position_prompt(1, "ABCDEFGHIJ"),
                         compact_prompt({"question": question, "context": "資料:ABCDEFGHIJ"}))

    def test_trained_position_5(self):
        self.assertIn(position_question(5), position_prompt(5, "ABCDEFGHIJ"))

    def test_trained_position_10(self):
        self.assertIn(position_question(10), position_prompt(10, "ABCDEFGHIJ"))

    def test_reject_unsupported_position(self):
        with self.assertRaisesRegex(ValueError, "1/5/10"):
            position_prompt(2, "ABCDEFGHIJ")

    def test_position_input_exact_10(self):
        with self.assertRaises(ValueError):
            position_prompt(5, "ABC")

    def test_parse_position_command(self):
        self.assertEqual(parse_position_command("/position 5 ABCDEFGHIJ"),
                         (5, "ABCDEFGHIJ"))

    def test_parse_position_non_int(self):
        with self.assertRaises(ValueError):
            parse_position_command("/position five ABCDEFGHIJ")

    def test_parse_position_short(self):
        with self.assertRaises(ValueError):
            parse_position_command("/position 5 ABC")

    def test_parse_position_missing(self):
        with self.assertRaises(ValueError):
            parse_position_command("/position 5")

    def test_copy_prompt_exact_training(self):
        self.assertEqual(copy_prompt("ABCDEFGHIJ"),
                         compact_prompt({"question": COPY_QUESTION,
                                         "context": "資料:ABCDEFGHIJ"}))

    def test_copy_reject_long(self):
        with self.assertRaises(ValueError):
            copy_prompt("ABCDEFGHIJK")

    def test_copy_reject_empty(self):
        with self.assertRaises(ValueError):
            copy_prompt("")

    def test_mode_names(self):
        self.assertEqual(CHAT_MODES, ("chat", "knowledge", "persona", "evidence"))

    def test_position_limit(self):
        self.assertEqual(POSITION_LIMIT, 10)

    def test_normal_prompt_unchanged(self):
        prompt, history = make_prompt("こんにちは", [], FakeTokenizer(), 256)
        self.assertEqual(prompt, compact_prompt(
            {"question": "こんにちは", "context": ""}))
        self.assertEqual(history, [])


if __name__ == "__main__":
    unittest.main()
