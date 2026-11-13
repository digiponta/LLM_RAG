import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluate_rope_generation_v094 import build_review, summarize, verify_paths


class RoPEGenerationV094Tests(unittest.TestCase):
    def setUp(self):
        self.p = dict(status="OK", text="はい", response_hash="a", chars=2,
                      eos=True, exact_match=False)
        self.r = dict(status="OK", text="そうですね", response_hash="b", chars=5,
                      eos=True, exact_match=False)

    def test_review_history_chronology(self):
        row = {"context": "寒い／服を着ます", "question": "あなたは？",
               "answer": "厚着です"}
        review = build_review(row, self.p, self.r, 2.5, 2.3)
        self.assertEqual(review["history"], ["寒い", "服を着ます"])
        self.assertEqual(review["question"], row["question"])
        self.assertEqual(review["legacy_response"], "はい")
        self.assertEqual(review["rope_response"], "そうですね")
        self.assertAlmostEqual(review["nll_gain"], .2)

    def test_review_has_no_fake_human_scores(self):
        row = {"context": "前文", "question": "質問", "answer": "回答"}
        review = build_review(row, self.p, self.r, 2, 2)
        for model in ("legacy", "rope"):
            self.assertTrue(all(value is None for value in
                                review["review_fields"][model].values()))

    def test_review_empty_history(self):
        row = {"context": "", "question": "質問", "answer": "回答"}
        self.assertEqual(build_review(row, self.p, self.r, 2, 2)["history"], [])

    def test_summary_empty(self):
        self.assertEqual(summarize([], 5), {"selected": 5, "usable": 0})

    def test_summary_positive(self):
        data = [
            {"nll_gain": .2, "response_changed": True,
             "legacy_generation": self.p, "rope_generation": self.r},
            {"nll_gain": -.1, "response_changed": False,
             "legacy_generation": self.p, "rope_generation": self.p},
        ]
        stats = summarize(data, 2)
        self.assertEqual(stats["usable"], 2)
        self.assertAlmostEqual(stats["mean_example_nll_gain"], .05)
        self.assertEqual(stats["rope_lower_nll_rate"], .5)
        self.assertEqual(stats["response_change_rate"], .5)

    def args(self, tmp, **kwargs):
        d = dict(sample=1, max_new_tokens=12, include_text=False,
                 baseline=Path(tmp)/"base.pt", rope_checkpoint=Path(tmp)/"rope.pt",
                 tokenizer=Path(tmp)/"tok.json", test_data=Path(tmp)/"test.jsonl",
                 output=Path(tmp)/"out.json", overwrite=False)
        d.update(kwargs)
        return SimpleNamespace(**d)

    def test_positive_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            verify_paths(self.args(tmp))

    def test_zero_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                verify_paths(self.args(tmp, sample=0))

    def test_zero_new_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                verify_paths(self.args(tmp, max_new_tokens=0))

    def test_private_filename_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                verify_paths(self.args(tmp, include_text=True))

    def test_private_filename_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            verify_paths(self.args(
                tmp, include_text=True,
                output=Path(tmp)/"report_text_v094.json"))

    def test_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                verify_paths(self.args(
                    tmp, output=Path(tmp)/"base.pt"))

    def test_existing_output_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"out.json"
            out.write_text("old")
            with self.assertRaises(FileExistsError):
                verify_paths(self.args(tmp))
            verify_paths(self.args(tmp, overwrite=True))


if __name__ == "__main__":
    unittest.main()
