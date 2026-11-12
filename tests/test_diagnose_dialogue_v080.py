"""CPU-only tests for the v0.8.0 decoding diagnostic metrics."""
import unittest
from diagnose_dialogue_v080 import (
    compact_prompt, repetition_stats, policy_parameters,
    aggregate, apply_repetition_penalty, PROBES,
)


class DialogueDiagnosticsV080Tests(unittest.TestCase):
    def test_prompt_format(self):
        self.assertEqual(compact_prompt("宇宙とは？"), "問:宇宙とは？\n答:")

    def test_empty_stats(self):
        self.assertEqual(repetition_stats([], "")["tokens"], 0)

    def test_repeated_token_run(self):
        stats = repetition_stats([1, 1, 1, 1, 2], "、、、、あ")
        self.assertEqual(stats["max_same_run"], 4)
        self.assertAlmostEqual(stats["adjacent_repeat_ratio"], 0.75)
        self.assertAlmostEqual(stats["unique_token_ratio"], 0.4)
        self.assertAlmostEqual(stats["punctuation_ratio"], 0.8)

    def test_unique_tokens(self):
        stats = repetition_stats([1, 2, 3, 4], "宇宙時間")
        self.assertEqual(stats["adjacent_repeat_ratio"], 0)
        self.assertEqual(stats["repeat_bigram_ratio"], 0)
        self.assertEqual(stats["unique_token_ratio"], 1.0)

    def test_repeated_bigram(self):
        stats = repetition_stats([1, 2, 1, 2, 1, 2], "abcdef")
        self.assertGreater(stats["repeat_bigram_ratio"], 0)

    def test_policy_parameters(self):
        self.assertEqual(policy_parameters("greedy")["temperature"], 0)
        self.assertEqual(policy_parameters("penalty")["penalty"], 1.3)
        self.assertGreater(policy_parameters("sample")["temperature"], 0)

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            policy_parameters("unknown")

    def test_aggregate(self):
        result = {"status": "OK", "eos_generated": True,
                  "generation_length": 2,
                  "repetition": {"adjacent_repeat_ratio": 0.0,
                                 "unique_token_ratio": 1.0,
                                 "punctuation_ratio": 0.0}}
        aggregated = aggregate([{"model": "tuned", "policy": "greedy", "result": result}])
        self.assertEqual(aggregated["tuned/greedy"]["eos_count"], 1)
        self.assertEqual(aggregated["tuned/greedy"]["cases"], 1)

    def test_probes_are_unique(self):
        self.assertEqual(len(PROBES), len({question for _, question in PROBES}))

    def test_torch_penalty_optional(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        logits = torch.tensor([2.0, -2.0, 0.0])
        changed = apply_repetition_penalty(logits, [0, 1], 2.0)
        self.assertAlmostEqual(changed[0].item(), 1.0)
        self.assertAlmostEqual(changed[1].item(), -4.0)
        self.assertAlmostEqual(changed[2].item(), 0.0)
        self.assertEqual(logits[0].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
