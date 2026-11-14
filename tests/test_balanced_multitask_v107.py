"""CPU-only tests for v0.10.7 balanced sampling and retention gates."""
import unittest

from prepare_balanced_multitask_v107 import (
    TASKS, WEIGHTS, quota, source_type, canonical_origin,
    sample_balanced, separation,
)
from evaluate_multitask_gate_v107 import decide


def row(task, index, split="train"):
    source = {
        "real_persona": ("real_persona_chat", "chat"),
        "nagato_knowledge": ("nagato_corpus", "source_attributed_qa"),
        "nagato_persona": ("nagato_corpus", "persona_style_from_source"),
        "evidence_binding": ("nagato_corpus", "evidence_direct"),
        "position": ("synthetic_composition", "position_extract"),
        "copy": ("synthetic_composition", "generalized_copy"),
    }
    kind, example = source[task]
    return {
        "id": f"{task}:{split}:{index}",
        "conversation_id": f"{task}:{split}:{index}",
        "question": "question", "context": f"資料:{task}-{split}-{index}",
        "answer": "はい", "source_kind": kind, "example_type": example,
    }


def report(ratio=1, accuracy=.7):
    tasks = {name: {"usable": 20, "models": {
        "binding_v100": {"answer_nll": 2},
        "multitask_v107": {"answer_nll": 2 * ratio},
    }} for name in ("nagato_knowledge", "nagato_persona",
                   "real_persona", "evidence_binding")}
    position = {"version": "v0.10.5", "groups": {"tuned": {
        key: {"cases": 50, "exact_rate": accuracy,
              "eos_termination_fraction": .95}
        for key in ("all", "position_1", "position_5",
                    "position_10", "natural", "synthetic")
    }}}
    return {"validation_only": True, "tasks": tasks}, position


class MultitaskV107Tests(unittest.TestCase):
    def setUp(self):
        self.groups = {task: [row(task, i) for i in range(20)] for task in TASKS}

    def test_weights_total(self):
        self.assertEqual(sum(WEIGHTS.values()), 100)

    def test_quota_4000(self):
        self.assertEqual(quota(4000), {
            "real_persona": 1600, "nagato_knowledge": 800,
            "nagato_persona": 600, "evidence_binding": 600,
            "position": 200, "copy": 200})

    def test_quota_400(self):
        self.assertEqual(sum(quota(400).values()), 400)

    def test_reject_total(self):
        with self.assertRaises(ValueError):
            quota(21)

    def test_reject_weights(self):
        with self.assertRaises(ValueError):
            quota(100, {**WEIGHTS, "position": 15})

    def test_task_filter(self):
        self.assertTrue(source_type("nagato_persona", row("nagato_persona", 1)))
        self.assertFalse(source_type("nagato_knowledge", row("nagato_persona", 1)))

    def test_reject_unknown_task(self):
        with self.assertRaises(ValueError):
            source_type("unknown", row("copy", 1))

    def test_balanced_sampling_deterministic(self):
        a, _ = sample_balanced(self.groups, 100, seed=1, max_repeat=8)
        b, _ = sample_balanced(self.groups, 100, seed=1, max_repeat=8)
        self.assertEqual(a, b)

    def test_balanced_counts(self):
        result, audit = sample_balanced(self.groups, 100, seed=1, max_repeat=8)
        self.assertEqual(len(result), 100)
        self.assertEqual(dict(__import__("collections").Counter(
            r["multitask"] for r in result)), quota(100))
        self.assertEqual(audit["real_persona"]["available_unique_rows"], 20)

    def test_refuse_heavy_repetition(self):
        with self.assertRaises(ValueError):
            sample_balanced(self.groups, 4000, max_repeat=8)

    def test_no_duplicate_output_ids(self):
        result, _ = sample_balanced(self.groups, 100, max_repeat=8)
        self.assertEqual(len({r["id"] for r in result}), len(result))

    def test_original_source_preserved(self):
        result, _ = sample_balanced(self.groups, 100, max_repeat=8)
        self.assertTrue(all(r["original_conversation_id"] for r in result))

    def test_separation_disjoint(self):
        train, _ = sample_balanced(self.groups, 100, max_repeat=8, split="train")
        dev_groups = {t: [row(t, i, "dev") for i in range(20)] for t in TASKS}
        dev, _ = sample_balanced(dev_groups, 100, max_repeat=8, split="validation")
        separation(train, dev)

    def test_separation_reject_shared(self):
        train, _ = sample_balanced(self.groups, 100, max_repeat=8, split="train")
        dev, _ = sample_balanced(self.groups, 100, max_repeat=8, split="validation")
        with self.assertRaises(ValueError):
            separation(train, dev)

    def test_original_line_through_copy(self):
        a = canonical_origin({"conversation_id": "position-v105:train:copy-v103:train:nagato-v099:xyz"})
        b = canonical_origin({"conversation_id": "nagato-v099:xyz"})
        self.assertEqual(a, b)

    def test_synthetic_origin_keeps_split(self):
        a = canonical_origin({"conversation_id": "copy-v103:train:synth-1"})
        b = canonical_origin({"conversation_id": "copy-v103:development:synth-1"})
        self.assertNotEqual(a, b)

    def test_gate_all_pass(self):
        a, b = report()
        self.assertEqual(decide(a, b)["status"], "DEVELOPMENT_GATE_PASS")
        self.assertFalse(decide(a, b)["accepted_for_deployment"])

    def test_gate_retention_reject(self):
        a, b = report(ratio=1.4)
        self.assertEqual(decide(a, b)["status"], "REJECT_REGRESSION")

    def test_gate_position_reject(self):
        a, b = report(accuracy=.1)
        self.assertEqual(decide(a, b)["status"], "REJECT_REGRESSION")

    def test_gate_needs_validation(self):
        a, b = report()
        a["validation_only"] = False
        with self.assertRaises(ValueError):
            decide(a, b)

    def test_gate_reject_wrong_position_report(self):
        a, b = report()
        b["version"] = "v0.10.1"
        with self.assertRaises(ValueError):
            decide(a, b)

    def test_gate_threshold_validation(self):
        a, b = report()
        with self.assertRaises(ValueError):
            decide(a, b, max_nll_ratio=.5)


if __name__ == "__main__":
    unittest.main()
