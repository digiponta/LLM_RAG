"""CPU-only validation for v0.10.6 multi-task retention grouping."""
import unittest
from pathlib import Path

from evaluate_nagato_retention_v106 import (
    TASKS, CHECKPOINTS, choose_rows, guard_validation_paths, sample_rows,
    summarize,
)


def row(i, kind, source):
    return {"id": f"{kind}:{i}", "conversation_id": f"src:{kind}:{i}",
            "example_type": kind, "source_kind": source,
            "question": "?", "answer": "回答です。"}


class RetentionV106Tests(unittest.TestCase):
    def setUp(self):
        self.nagato = [
            row(1, "source_attributed_qa", "nagato_corpus"),
            row(2, "persona_style_from_source", "nagato_corpus"),
        ]
        self.unified = [
            row(3, "anything", "real_persona_chat"),
            row(4, "source_attributed_qa", "nagato_corpus"),
        ]
        self.binding = [
            row(5, "evidence_direct", "nagato_corpus"),
            row(6, "evidence_followup", "nagato_corpus"),
        ]

    def test_four_groups(self):
        grouped = choose_rows(self.nagato, self.unified, self.binding)
        self.assertEqual(set(grouped), set(TASKS))
        self.assertEqual([len(grouped[k]) for k in TASKS], [1, 1, 1, 1])

    def test_only_attributed_knowledge(self):
        n = self.nagato + [row(7, "random", "nagato_corpus")]
        self.assertEqual(len(choose_rows(n, self.unified, self.binding)["nagato_knowledge"]), 1)

    def test_no_unverified_answers_promoted(self):
        group = choose_rows(self.nagato, self.unified, self.binding)
        self.assertEqual(group["nagato_knowledge"][0]["source_kind"], "nagato_corpus")

    def test_no_binding_followup(self):
        group = choose_rows(self.nagato, self.unified, self.binding)
        self.assertEqual(group["evidence_binding"][0]["example_type"], "evidence_direct")

    def test_real_persona_filters_nagato(self):
        group = choose_rows(self.nagato, self.unified, self.binding)
        self.assertEqual(group["real_persona"][0]["id"], "anything:3")

    def test_missing_group_fails(self):
        with self.assertRaises(ValueError):
            choose_rows(self.nagato, self.unified, [])

    def test_duplicate_group_id_fails(self):
        with self.assertRaises(ValueError):
            choose_rows([*self.nagato, self.nagato[0]], self.unified, self.binding)

    def test_reject_test_filename(self):
        with self.assertRaises(ValueError):
            guard_validation_paths([Path("data/nagato_test_v099.jsonl")])

    def test_reject_holdout_filename(self):
        with self.assertRaises(ValueError):
            guard_validation_paths([Path("data/final_holdout.jsonl")])

    def test_reject_same_file(self):
        with self.assertRaises(ValueError):
            guard_validation_paths([Path("a.jsonl"), Path("a.jsonl")])

    def test_allow_validation_files(self):
        guard_validation_paths([Path("a_validation.jsonl"), Path("b_dev.jsonl")])

    def test_seeded_subset_deterministic(self):
        data = [row(i, "source_attributed_qa", "nagato_corpus") for i in range(20)]
        a = sample_rows(data, 5, 42)
        b = sample_rows(data, 5, 42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5)

    def test_sample_larger_than_source(self):
        self.assertEqual(len(sample_rows(self.nagato, 9, 42)), 2)

    def test_sample_zero_rejected(self):
        with self.assertRaises(ValueError):
            sample_rows(self.nagato, 0, 42)

    def test_summarize_empty(self):
        a = summarize([], CHECKPOINTS)
        self.assertEqual(a["binding_v100"]["generated_samples"], 0)
        self.assertIsNone(a["binding_v100"]["exact_response_rate"])

    def test_summarize_generations(self):
        samples = [
            {"reference_chars": 5, "binding_v100": {
                "exact_match": True, "ended_by_eos": True, "generated_chars": 5}},
            {"reference_chars": 6, "binding_v100": {
                "exact_match": False, "ended_by_eos": False, "generated_chars": 4}},
        ]
        summary = summarize(samples, ["binding_v100"])["binding_v100"]
        self.assertEqual(summary["exact_response_rate"], .5)
        self.assertEqual(summary["eos_termination_fraction"], .5)
        self.assertEqual(summary["mean_answer_length_abs_error"], 1)


if __name__ == "__main__":
    unittest.main()
