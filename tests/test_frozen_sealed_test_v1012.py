"""Preflight and provenance regression tests. NEVER open sealed TEST on CI."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluate_frozen_sealed_test_v1012 import (
    audit_partition, origin, preflight, _task_rows)


def row(id_, conversation, question="問", answer="答",
        source_kind="nagato_corpus", example_type="source_attributed_qa"):
    return {
        "id": id_, "conversation_id": conversation,
        "question": question, "answer": answer, "context": "",
        "source_kind": source_kind, "example_type": example_type,
    }


class ProvenanceTests(unittest.TestCase):
    def test_original_source_unwrap(self):
        item = row("one", "balanced-v107:train:nagato_knowledge:nagato-v099:abc")
        item["original_conversation_id"] = "nagato-v099:abc"
        self.assertEqual(origin(item), "nagato-v099:abc")

    def test_origin_nested_unwrap(self):
        item = row("one", "position-v105:train:copy-v103:train:nagato-v099:abc")
        self.assertEqual(origin(item), "nagato-v099:abc")

    def test_source_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "origin"):
            audit_partition([row("train", "nagato-v099:A")],
                            [row("test", "nagato-v099:A", answer="different")])

    def test_id_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "ID"):
            audit_partition([row("same", "A")], [row("same", "B")])

    def test_exact_qa_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "Q/A"):
            audit_partition([row("a", "A")], [row("b", "B")])

    def test_disjoint_passes(self):
        status = audit_partition([row("a", "A")],
                                 [row("b", "B", answer="different")])
        self.assertEqual(status["origin_overlap"], 0)
        self.assertEqual(status["test_rows"], 1)

    def test_duplicate_test_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside sealed"):
            audit_partition([], [row("x", "X"), row("x", "Y")])

    def test_groups_select_exact_types(self):
        nagato = [row("a", "A"), row("b", "B",
                  example_type="persona_style_from_source")]
        binding = [row("c", "C", example_type="evidence_direct")]
        groups = _task_rows(nagato, binding, None)
        self.assertEqual(len(groups["nagato_knowledge"]), 1)
        self.assertEqual(len(groups["nagato_persona"]), 1)
        self.assertEqual(len(groups["evidence_binding"]), 1)
        self.assertNotIn("real_persona", groups)

    def test_persona_optional(self):
        groups = _task_rows([row("a", "A")], [row("c", "C",
              example_type="evidence_direct")],
              [row("p", "P", source_kind="real_persona_chat")])
        self.assertEqual(len(groups["real_persona"]), 1)


class NoTestReadTests(unittest.TestCase):
    def test_preflight_does_not_open_test_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "candidate.pt"
            baseline = root / "baseline.pt"
            tokenizer = root / "tokenizer.json"
            train = root / "train.jsonl"
            for path, body in [(checkpoint, b"candidate"), (baseline, b"base"),
                               (tokenizer, b"{}"), (train, b"")]:
                path.write_bytes(body)
            args = SimpleNamespace(
                checkpoint=checkpoint, baseline=baseline, tokenizer=tokenizer,
                train_mixed=train, report=root / "report.json",
                nagato_test=root / "NOT_PRESENT_nagato_test.jsonl",
                binding_test=root / "NOT_PRESENT_binding_test.jsonl",
                persona_test=root / "NOT_PRESENT_persona_test.jsonl",
                max_rows=0, generations=12, max_new_tokens=96,
                batch_size=16, max_unk_fraction=.1,
            )
            result = preflight(args)
            self.assertFalse(result["opened_final_test"])
            self.assertEqual(result["status"], "READY_TO_OPEN_SEALED_TEST")
            self.assertFalse(args.report.exists())
            self.assertEqual(len(result["requested_test_paths_not_opened"]), 3)

    def test_overwrite_preflight_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / x for x in ("candidate.pt", "baseline.pt",
                                      "tokenizer.json", "train.jsonl", "report.json")]
            for p in paths:
                p.write_text("z")
            args = SimpleNamespace(
                checkpoint=paths[0], baseline=paths[1], tokenizer=paths[2],
                train_mixed=paths[3], report=paths[4],
                nagato_test=root / "nagato_test.jsonl",
                binding_test=root / "binding_test.jsonl", persona_test=None,
                max_rows=0, generations=12, max_new_tokens=96,
                batch_size=16, max_unk_fraction=.1,
            )
            with self.assertRaises(FileExistsError):
                preflight(args)


if __name__ == "__main__":
    unittest.main()
