"""v0.9.8 first-token and gold-prefix decoding tests.

Synthetic tiny checkpoints and tokenizers only; no private dialogue files.
"""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

import torch

from evaluate_generation_diagnostics_v098 import (
    first_token_diagnostic, prefix_continuation_diagnostic, summarize,
)
from rope_model_v093 import RoPELanguageModel


class StubTokenizer:
    eos_id = 2
    def decode(self, ids, skip_special_tokens=True):
        return "|".join(str(x) for x in ids if x != self.eos_id)


class StubModel(torch.nn.Module):
    def __init__(self, continuation, context_length=64):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.context_length = context_length
        self.continuation = continuation
        self.last_priming = None

    def forward(self, x):
        b, t = x.shape
        logits = torch.zeros((b, t, 16), device=x.device)
        logits[:, -1, 5] = 8.0
        return logits

    def generate(self, ids, **kwargs):
        self.last_priming = list(ids)
        return list(ids) + list(self.continuation) + [2]


class GenerationDiagnosticsTests(unittest.TestCase):
    def test_first_token_correct(self):
        out = first_token_diagnostic(StubModel([6]), [1, 3], [5, 6])
        self.assertTrue(out["correct"])
        self.assertEqual(out["target_rank"], 1)
        self.assertGreater(out["target_probability"], .90)

    def test_first_token_wrong(self):
        out = first_token_diagnostic(StubModel([]), [1], [4])
        self.assertFalse(out["correct"])
        self.assertGreater(out["target_rank"], 1)

    def test_empty_reference_rejected(self):
        with self.assertRaises(ValueError):
            first_token_diagnostic(StubModel([]), [1], [])

    def test_context_limit(self):
        with self.assertRaises(ValueError):
            first_token_diagnostic(StubModel([], context_length=2), [1, 2], [5])

    def test_gold_prefix_only_primes_expected_tokens(self):
        model = StubModel([7, 8])
        item = prefix_continuation_diagnostic(model, StubTokenizer(),
                                               [1, 3], [5, 6, 7, 8], 2)
        self.assertEqual(model.last_priming, [1, 3, 5, 6])
        self.assertEqual(item["reference_prefix"], "5|6")
        self.assertTrue(item["continuation_exact"])
        self.assertEqual(item["matched_leading_tokens"], 2)

    def test_prefix_not_counted_in_match(self):
        model = StubModel([7])
        result = prefix_continuation_diagnostic(model, StubTokenizer(),
                                                 [1], [5, 6, 7, 8], 2)
        self.assertFalse(result["continuation_exact"])
        self.assertEqual(result["leading_token_coverage"], .5)

    def test_gold_prefix_one(self):
        x = prefix_continuation_diagnostic(StubModel([6]), StubTokenizer(),
                                            [1], [5, 6], 1)
        self.assertTrue(x["continuation_exact"])

    def test_prefix_zero_rejected(self):
        with self.assertRaises(ValueError):
            prefix_continuation_diagnostic(StubModel([]), StubTokenizer(),
                                            [1], [5, 6], 0)

    def test_prefix_too_long_rejected(self):
        with self.assertRaises(ValueError):
            prefix_continuation_diagnostic(StubModel([]), StubTokenizer(),
                                            [1], [5, 6], 2)

    def test_priming_context_rejected(self):
        with self.assertRaises(ValueError):
            prefix_continuation_diagnostic(StubModel([], context_length=3),
                                            StubTokenizer(), [1, 2], [5, 6], 1)

    def test_bad_generation_limit(self):
        with self.assertRaises(ValueError):
            prefix_continuation_diagnostic(StubModel([]), StubTokenizer(),
                                            [1], [5, 6], 1, generation_limit=0)

    def test_summary(self):
        first = {"correct": True, "target_rank": 1, "target_nll": 0.1}
        row = {"first_token": first,
               "prefix": {"1": {"continuation_exact": True,
                                "leading_token_coverage": 1.0}}}
        out = summarize([row, row])
        self.assertEqual(out["first_token_accuracy"], 1.0)
        self.assertEqual(out["prefix"]["1"]["cases"], 2)

    def test_empty_summary(self):
        self.assertEqual(summarize([]), {"count": 0})

    def test_real_rope_first_token_cpu(self):
        torch.manual_seed(123)
        model = RoPELanguageModel(vocab_size=16, d_model=8, num_layers=1,
                                  hidden_dim=16, causal=True, context_length=16)
        result = first_token_diagnostic(model, [1, 3], [4, 5])
        self.assertGreaterEqual(result["target_rank"], 1)
        self.assertLessEqual(result["target_rank"], 16)


if __name__ == "__main__":
    unittest.main()
