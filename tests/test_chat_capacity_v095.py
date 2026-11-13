"""v0.9.5 tests: chat CLI prompt safety and model capacity configs."""
import unittest

from chat import make_prompt, resolve_rope
from initialize_capacity_v095 import PROFILES, build_config


class CharTokenizerStub:
    def encode(self, text, add_bos=False):
        return ([1] if add_bos else []) + [ord(c) for c in text]


class ChatV095Tests(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizerStub()

    def test_first_question_prompt(self):
        prompt, history = make_prompt("こんにちは", [], self.tok, 256)
        self.assertEqual(prompt, "問:こんにちは\n答:")
        self.assertEqual(history, [])

    def test_history_uses_latest_two(self):
        prompt, used = make_prompt("あなたは？", ["古い", "前A", "前B"], self.tok, 256)
        self.assertEqual(used, ["前A", "前B"])
        self.assertEqual(prompt, "前:前A／前B\n問:あなたは？\n答:")

    def test_overlength_history_dropped(self):
        prompt, used = make_prompt("質問", ["履歴" * 100], self.tok, 64)
        self.assertEqual(used, [])
        self.assertEqual(prompt, "問:質問\n答:")

    def test_question_never_truncated(self):
        with self.assertRaisesRegex(ValueError, "Question too long"):
            make_prompt("質問" * 100, [], self.tok, 64)

    def test_whitespace_question_rejected(self):
        with self.assertRaises(ValueError):
            make_prompt("  ", [], self.tok, 64)

    def test_invalid_max_new(self):
        with self.assertRaises(ValueError):
            make_prompt("こんにちは", [], self.tok, 64, max_new_tokens=0)

    def test_evidence_kept_if_fits(self):
        prompt, _ = make_prompt("宝石", [], self.tok, 256, evidence="鉱物")
        self.assertIn("資料:鉱物", prompt)

    def test_evidence_shrinks_without_truncating_question(self):
        question = "詳しく"
        prompt, _ = make_prompt(question, [], self.tok, 40, evidence="鉱物" * 80)
        self.assertIn(f"問:{question}\n答:", prompt)
        self.assertLess(len(self.tok.encode(prompt, add_bos=True)), 40)

    def test_auto_rope(self):
        self.assertTrue(resolve_rope("auto", {"position_encoding": "rope_v093"}))
        self.assertFalse(resolve_rope("auto", {}))

    def test_force_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            resolve_rope("legacy", {"position_encoding": "rope_v093"})
        with self.assertRaises(ValueError):
            resolve_rope("rope", {})

    def test_unknown_encoding_rejected(self):
        with self.assertRaises(ValueError):
            resolve_rope("auto", {"position_encoding": "invalid"})


class CapacityV095Tests(unittest.TestCase):
    def test_profiles_valid(self):
        for name in PROFILES:
            config = build_config(name, 5117, 256)
            self.assertEqual(config["vocab_size"], 5117)
            self.assertEqual(config["context_length"], 256)
            self.assertEqual(config["causal"], True)
            self.assertEqual(config["d_model"] % 2, 0)

    def test_profiles_scale(self):
        self.assertLess(PROFILES["small"]["d_model"], PROFILES["medium"]["d_model"])
        self.assertLess(PROFILES["medium"]["num_layers"], PROFILES["large"]["num_layers"])

    def test_unknown_profile(self):
        with self.assertRaises(ValueError):
            build_config("unknown", 5117, 256)

    def test_bad_vocab_context(self):
        with self.assertRaises(ValueError):
            build_config("small", 3, 256)
        with self.assertRaises(ValueError):
            build_config("small", 5117, 16)


if __name__ == "__main__":
    unittest.main()
