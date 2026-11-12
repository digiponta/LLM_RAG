"""v0.8.6 diagnostics tests; pure Python tests do not require CUDA."""
import unittest

from diagnose_generation_v086 import config_for, hash_text, summarize, POLICIES


def case(answer, eos=True, adjacent=0.0, bigram=0.0):
    return {
        "status": "OK", "response_hash": hash_text(answer),
        "response_chars": len(answer), "eos": eos, "generated_tokens": len(answer),
        "repetition": {"adjacent_repeat_ratio": adjacent, "repeat_bigram_ratio": bigram},
    }


class GenerationDiagnosticsV086Tests(unittest.TestCase):
    def test_policies_present(self):
        self.assertEqual(set(POLICIES), {"greedy", "penalty", "sample", "topk", "topk_penalty"})

    def test_greedy_configuration(self):
        self.assertEqual(config_for("greedy")["temperature"], 0.0)
        self.assertEqual(config_for("greedy")["penalty"], 1.0)

    def test_penalty_configuration(self):
        self.assertEqual(config_for("penalty")["penalty"], 1.3)

    def test_topk_configuration(self):
        self.assertEqual(config_for("topk")["top_k"], 20)

    def test_unrestricted_sample_configuration(self):
        self.assertIsNone(config_for("sample")["top_k"])

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            config_for("bogus")

    def test_summary_empty(self):
        self.assertEqual(summarize([]), {"cases": 0})

    def test_summary_no_valid(self):
        self.assertEqual(summarize([{"status": "PROMPT_TOO_LONG"}]),
                         {"cases": 1, "valid": 0})

    def test_common_response_detected(self):
        s = summarize([case("そうですね"), case("そうですね"), case("はい")])
        self.assertEqual(s["unique_response_ratio"], 2 / 3)
        self.assertEqual(s["most_common_response_share"], 2 / 3)

    def test_eos_rate(self):
        s = summarize([case("はい"), case("こんにちは", eos=False)])
        self.assertEqual(s["eos_rate"], .5)

    def test_skip_and_summary(self):
        s = summarize([case("はい"), {"status": "HIGH_PROMPT_UNK"}])
        self.assertEqual(s["cases"], 2)
        self.assertEqual(s["valid"], 1)

    def test_repeat_stats(self):
        s = summarize([case("ああ", adjacent=.5, bigram=.25), case("はい")])
        self.assertAlmostEqual(s["mean_adjacent_repeat_ratio"], .25)
        self.assertAlmostEqual(s["mean_repeat_bigram_ratio"], .125)

    def test_hash_stable_and_private(self):
        h = hash_text("秘密の対話")
        self.assertEqual(h, hash_text("秘密の対話"))
        self.assertEqual(len(h), 16)
        self.assertNotIn("秘密", h)

    def test_status_shape(self):
        self.assertEqual(summarize([case(""), case("はい")])["empty_rate"], .5)


if __name__ == "__main__":
    unittest.main()
