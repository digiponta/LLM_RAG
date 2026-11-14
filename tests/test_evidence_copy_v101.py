"""v0.10.1 evidence-copy diagnostics on synthetic records."""
import unittest

from evaluate_evidence_copy_v101 import (
    assemble_cases, copy_row, longest_common_prefix,
    score_copy, summarize,
)


class StubTokenizer:
    unk_id = 3
    eos_id = 2

    def encode(self, text, add_bos=False):
        return ([1] if add_bos else []) + [10 + (ord(ch) % 60) for ch in text]


def source(i, text=None):
    quote = text if text is not None else "宇宙と時間の観測によって原文を記録します。今後の検証が必要です。"
    return {
        "id": f"nagato:{i}:source",
        "conversation_id": f"nagato-line:{i}",
        "question": "原文は？",
        "answer": f"長門コーパスには「{quote}」と記されています。",
        "context": "",
        "example_type": "source_attributed_qa",
        "source_kind": "nagato_corpus",
    }


class CopyV101Tests(unittest.TestCase):
    def test_exact_copy_short(self):
        row = copy_row(source(1), 10)
        self.assertEqual(row["answer"], "宇宙と時間の観測によ")
        self.assertEqual(row["context"], "資料:" + row["answer"])

    def test_quote_not_overlength(self):
        self.assertIsNone(copy_row(source(1, "短い文。"), 10))

    def test_positive_width(self):
        with self.assertRaises(ValueError):
            copy_row(source(1), 0)

    def test_reject_non_attributed(self):
        self.assertIsNone(copy_row({**source(1), "example_type": "persona_style_from_source"}, 10))

    def test_sft_encoder_accepts_copy(self):
        from train_real_dialogue_v082 import encode_dialogue
        pair, reason = encode_dialogue(copy_row(source(1), 10), StubTokenizer(), 256)
        self.assertEqual(reason, "accepted")
        self.assertEqual(len(pair[0]), len(pair[1]))

    def test_case_selection_reproducible(self):
        rows = [source(i) for i in range(25)]
        a, _ = assemble_cases(rows, StubTokenizer(), widths=[10, 30],
                              cases_per_width=3, context_length=256,
                              max_unk_fraction=.1)
        b, _ = assemble_cases(rows, StubTokenizer(), widths=[10, 30],
                              cases_per_width=3, context_length=256,
                              max_unk_fraction=.1)
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])

    def test_case_selection_no_source_repeat(self):
        rows = [source(i) for i in range(30)]
        cases, _ = assemble_cases(rows, StubTokenizer(), widths=[10, 30, 50],
                                  cases_per_width=2, context_length=256,
                                  max_unk_fraction=.1)
        self.assertEqual(len(cases), len(set(r["conversation_id"] for r in cases)))

    def test_no_cases_rejected(self):
        with self.assertRaises(ValueError):
            assemble_cases([source(1, "短い文。")], StubTokenizer(), widths=[50],
                           cases_per_width=1, context_length=256, max_unk_fraction=.1)

    def test_empty_widths_rejected(self):
        with self.assertRaises(ValueError):
            assemble_cases([], StubTokenizer(), widths=[],
                           cases_per_width=1, context_length=256, max_unk_fraction=.1)

    def test_duplicate_widths_rejected(self):
        with self.assertRaises(ValueError):
            assemble_cases([], StubTokenizer(), widths=[10, 10],
                           cases_per_width=1, context_length=256, max_unk_fraction=.1)

    def test_invalid_sample_count(self):
        with self.assertRaises(ValueError):
            assemble_cases([], StubTokenizer(), widths=[10],
                           cases_per_width=0, context_length=256, max_unk_fraction=.1)

    def test_prefix_length(self):
        self.assertEqual(longest_common_prefix("宇宙時間", "宇宙空間"), 2)

    def test_score_copy_exact(self):
        result = score_copy("宇宙時間", "宇宙時間", eos=True)
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["prefix_coverage"], 1)
        self.assertEqual(result["position_accuracy"], 1)
        self.assertEqual(result["sequence_similarity"], 1)

    def test_score_copy_mismatch(self):
        result = score_copy("宇宙時間", "宇宙空間", eos=False)
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["prefix_chars"], 2)
        # 宇宙時間 vs 宇宙空間: 宇/宙/間 match by position (3/4),
        # while contiguous common-prefix coverage remains 2/4.
        self.assertEqual(result["position_accuracy"], .75)
        self.assertEqual(result["prefix_coverage"], .5)

    def test_score_copy_no_answer(self):
        result = score_copy("宇宙", "", eos=False)
        self.assertEqual(result["prefix_coverage"], 0)
        self.assertEqual(result["length_error"], -2)

    def test_empty_reference(self):
        with self.assertRaises(ValueError):
            score_copy("", "宇宙", eos=False)

    def test_score_summary(self):
        metric = score_copy("宇宙", "宇宙", eos=True)
        metric.update(first_token_correct=True, first_token_rank=1)
        case = {"length_chars": 10, "baseline": metric}
        result = summarize([case], "baseline", [10, 30])
        self.assertEqual(result["10"]["exact_rate"], 1)
        self.assertEqual(result["30"], {"cases": 0})


if __name__ == "__main__":
    unittest.main()
