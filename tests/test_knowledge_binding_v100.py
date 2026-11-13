"""v0.10.0 evidence-binding tests on synthetic text only."""
import tempfile
import unittest
from pathlib import Path

from prepare_knowledge_binding_v100 import (
    bound_variants, check_partitions, excerpt_from_v099,
    no_evidence, prepare_bound,
)
from train_real_dialogue_v082 import encode_dialogue


class CharTokenizer:
    eos_id = 2
    unk_id = 3
    def encode(self, text, add_bos=False):
        return ([1] if add_bos else []) + [4 + ord(ch) % 40 for ch in text]


def source(index=1, text="宇宙の時間について考えます。", kind="source_attributed_qa"):
    return {
        "id": f"n{index}:source",
        "conversation_id": f"line-{index}",
        "source_kind": "nagato_corpus",
        "example_type": kind,
        "answer": f"長門コーパスには「{text}」と記されています。",
        "question": "原文の記述は？",
        "context": "",
    }


class BindingV100Tests(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()

    def test_exact_excerpt(self):
        self.assertEqual(excerpt_from_v099(source()), "宇宙の時間について考えます。")

    def test_reject_forged_envelope(self):
        row = source()
        row["answer"] = "宇宙とは世界の全体です"
        with self.assertRaises(ValueError):
            excerpt_from_v099(row)

    def test_reject_persona_as_provenance(self):
        with self.assertRaises(ValueError):
            excerpt_from_v099(source(kind="persona_style_from_source"))

    def test_direct_and_followup_share_source(self):
        rows = bound_variants(source())
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["conversation_id"] for row in rows}, {"line-1"})
        self.assertEqual(len({row["id"] for row in rows}), 2)

    def test_evidence_is_verbatim(self):
        for row in bound_variants(source()):
            self.assertIn("宇宙の時間について考えます。", row["context"])
            self.assertEqual(row["answer"],
                             "資料には「宇宙の時間について考えます。」と記されています。")

    def test_unknown_topic_skipped(self):
        self.assertEqual(bound_variants(source(text="砂糖と胡椒を混ぜます。")), [])

    def test_long_excerpt_truncated_only_at_prefix(self):
        text = "宇宙" + "時" * 100
        rows = bound_variants(source(text=text), excerpt_chars=20)
        self.assertEqual(rows[0]["answer"], "資料には「" + text[:20] + "」と記されています。")

    def test_tiny_excerpt_rejected(self):
        with self.assertRaises(ValueError):
            bound_variants(source(), excerpt_chars=8)

    def test_shared_sft_mask_compatible(self):
        rows, stats = prepare_bound([source()], self.tok, context_length=256)
        self.assertGreaterEqual(len(rows), 1)
        for row in rows:
            encoded, status = encode_dialogue(row, self.tok, 256)
            self.assertEqual(status, "accepted")
            self.assertEqual(len(encoded[0]), len(encoded[1]))

    def test_skip_unqualified_source(self):
        rows, _ = prepare_bound([source(kind="persona_style_from_source")], self.tok)
        self.assertEqual(rows, [])

    def test_no_evidence_removes_quote_and_keeps_answer(self):
        originals, _ = prepare_bound([source()], self.tok)
        control = no_evidence(originals, self.tok)
        self.assertEqual(len(originals), len(control))
        for src, row in zip(originals, control):
            self.assertEqual(row["answer"], src["answer"])
            self.assertNotIn("宇宙の時間について考えます。", row["context"])
            self.assertNotEqual(row["id"], src["id"])

    def test_no_evidence_direct_empty(self):
        rows = bound_variants(source())
        control = no_evidence([rows[0]], self.tok)
        self.assertEqual(control[0]["context"], "")

    def test_no_evidence_followup_retains_topic_not_answer(self):
        rows = bound_variants(source())
        control = no_evidence([rows[1]], self.tok)
        self.assertIn("話題:宇宙", control[0]["context"])
        self.assertNotIn("資料:", control[0]["context"])

    def test_line_leakage_rejected(self):
        original = bound_variants(source())[0]
        with self.assertRaises(ValueError):
            check_partitions([original], [{**original, "id": "other"}], [])

    def test_duplicate_ids_rejected(self):
        original = bound_variants(source())[0]
        with self.assertRaises(ValueError):
            check_partitions([original, original], [], [])

    def test_disjoint_splits_allowed(self):
        check_partitions(bound_variants(source(1)), bound_variants(source(2)),
                         bound_variants(source(3)))

    def test_source_quote_is_not_verified_truth(self):
        rows = bound_variants(source(text="宇宙は泡かもしれない。"))
        self.assertEqual(rows[0]["review_state"], "SOURCE_QUOTE_UNVERIFIED")
        self.assertIn("かもしれない", rows[0]["answer"])


if __name__ == "__main__":
    unittest.main()
