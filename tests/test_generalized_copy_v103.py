"""v0.10.3 source-disjoint compositional copy dataset and scorer tests."""
import unittest

from prepare_generalized_copy_v103 import (
    originals, alphabet, make_row, choose, guard,
)
from evaluate_generalized_copy_v103 import validate_copy_rows, summarize
from evaluate_evidence_copy_v101 import score_copy


class StubTokenizer:
    unk_id = 3
    eos_id = 2

    def encode(self, text, add_bos=False):
        return ([1] if add_bos else []) + [4 + (ord(ch) % 80) for ch in text]


def source(i, text=None):
    quote = text or "宇宙の時間と物質の関係は、観測者の状態によって異なります。"
    return {
        "id": f"n{i}:source",
        "conversation_id": f"original:{i}",
        "example_type": "source_attributed_qa",
        "source_kind": "nagato_corpus",
        "question": "原文は",
        "context": "",
        "answer": f"長門コーパスには「{quote}」と記されています。",
    }


class GeneralizedCopyV103Tests(unittest.TestCase):
    def setUp(self):
        self.tok = StubTokenizer()
        self.train_source = originals([source(i) for i in range(30)])
        self.dev_source = originals([source(i) for i in range(30, 40)])
        self.pool = alphabet(self.train_source)

    def selected(self):
        return choose(self.train_source, self.pool, self.tok,
                      split="train", count=20, width=10)

    def test_extracts_only_attributed(self):
        normal = source(1)
        excluded = {**source(2), "example_type": "persona_style_from_source"}
        self.assertEqual(len(originals([normal, excluded])), 1)

    def test_alphabet_from_training(self):
        self.assertIn("宇", self.pool)
        self.assertNotIn("X", self.pool)

    def test_alphabet_too_small(self):
        with self.assertRaises(ValueError):
            alphabet([("x", "あ")])

    def test_generator_count(self):
        rows, stats = self.selected()
        self.assertEqual(len(rows), 20)
        self.assertEqual(stats["accepted"], 20)

    def test_generator_deterministic(self):
        a, _ = self.selected()
        b, _ = self.selected()
        self.assertEqual([x["answer"] for x in a], [x["answer"] for x in b])

    def test_generator_both_kinds(self):
        rows, _ = self.selected()
        self.assertEqual({r["copy_kind"] for r in rows}, {"natural", "synthetic"})

    def test_answer_matches_evidence(self):
        rows, _ = self.selected()
        self.assertTrue(all(r["context"] == "資料:" + r["answer"] for r in rows))

    def test_no_repeated_answers(self):
        rows, _ = self.selected()
        self.assertEqual(len({r["answer"] for r in rows}), 20)

    def test_development_disjoint(self):
        train, _ = self.selected()
        dev, _ = choose(self.dev_source, self.pool, self.tok, split="development",
                        count=20, width=10, seed=43,
                        banned={r["answer"] for r in train})
        guard(train, dev)

    def test_leaked_answers_rejected(self):
        train, _ = self.selected()
        dev = [{**train[0], "id": "different", "conversation_id": "other"}]
        with self.assertRaises(ValueError):
            guard(train, dev)

    def test_leaked_source_origin_rejected(self):
        a = make_row("宇宙時間量子法則放送", split="train",
                     kind="natural", origin="same", index=1)
        b = make_row("量子計算と音声取得関係", split="train",
                     kind="natural", origin="same", index=2)
        with self.assertRaises(ValueError):
            guard([a], [b])

    def test_too_short_natural_rejected(self):
        with self.assertRaises(ValueError):
            choose([("short", "短い")], self.pool, self.tok, split="train",
                   count=1, width=10)

    def test_nonpositive_count_rejected(self):
        with self.assertRaises(ValueError):
            choose(self.train_source, self.pool, self.tok,
                   split="train", count=0)

    def test_nonpositive_width_rejected(self):
        with self.assertRaises(ValueError):
            choose(self.train_source, self.pool, self.tok,
                   split="train", count=1, width=0)

    def test_eval_rejects_forged_target(self):
        a = make_row("時間と物質である", split="train",
                     kind="synthetic", origin="x", index=1)
        a["answer"] = "違う文章"
        with self.assertRaises(ValueError):
            validate_copy_rows([a])

    def test_eval_accepts_matched_rows(self):
        rows, _ = self.selected()
        validate_copy_rows(rows)

    def test_eval_rejects_duplicate(self):
        rows, _ = self.selected()
        with self.assertRaises(ValueError):
            validate_copy_rows([rows[0], rows[0]])

    def test_eval_summary_separate_kinds(self):
        m = score_copy("宇宙", "宇宙", eos=True)
        m["first_token_correct"] = True
        entries = [{"kind": "natural", "baseline": m},
                   {"kind": "synthetic", "baseline": m}]
        summary = summarize(entries, "baseline")
        self.assertEqual(summary["natural"]["cases"], 1)
        self.assertEqual(summary["synthetic"]["cases"], 1)
        self.assertEqual(summary["all"]["cases"], 2)


if __name__ == "__main__":
    unittest.main()
