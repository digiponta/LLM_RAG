"""v0.10.2 controlled copy learning probe tests (no model weights required)."""
import unittest

from prepare_copy_learning_v102 import guard_sources, select_examples
from train_copy_learning_v102 import validate_partitions


class StubTokenizer:
    unk_id = 3
    eos_id = 2

    def encode(self, text, add_bos=False):
        return ([1] if add_bos else []) + [10 + ord(ch) % 60 for ch in text]


def row(i, text="物質波は、ドブロイ波である。物理学に関する文章です。"):
    return {
        "id": f"nagato:{i}:source",
        "conversation_id": f"nagato-line:{i}",
        "question": "原文は？",
        "answer": f"長門コーパスには「{text}」と記されています。",
        "context": "",
        "source_kind": "nagato_corpus",
        "example_type": "source_attributed_qa",
    }


class CopyLearningV102Tests(unittest.TestCase):
    def setUp(self):
        self.tok = StubTokenizer()

    def test_reproducible_selection(self):
        source = [row(i) for i in range(30)]
        a = select_examples(source, self.tok, count=4)
        b = select_examples(source, self.tok, count=4)
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])

    def test_experiment_data_is_literal(self):
        chosen = select_examples([row(1)], self.tok, count=1)
        self.assertEqual(chosen[0]["answer"], chosen[0]["context"][3:])
        self.assertEqual(len(chosen[0]["answer"]), 10)

    def test_select_unique_origins(self):
        chosen = select_examples([row(1), row(1), row(2)], self.tok, count=3)
        self.assertEqual(len(chosen), 2)

    def test_requested_count_respected(self):
        chosen = select_examples([row(i) for i in range(20)], self.tok, count=4)
        self.assertEqual(len(chosen), 4)

    def test_skip_short_source(self):
        selected = select_examples([row(1, "短い。"), row(2)], self.tok, count=1)
        self.assertEqual(selected[0]["conversation_id"], "nagato-line:2")

    def test_reject_empty_pool(self):
        with self.assertRaises(ValueError):
            select_examples([], self.tok)

    def test_bad_width(self):
        with self.assertRaises(ValueError):
            select_examples([row(1)], self.tok, width=0)

    def test_no_source_overlap(self):
        guard_sources([{"conversation_id": "A"}], [{"conversation_id": "B"}])

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            guard_sources([{"conversation_id": "A"}], [{"conversation_id": "A"}])

    def test_duplicate_origins_rejected(self):
        with self.assertRaises(ValueError):
            guard_sources([{"conversation_id": "A"}, {"conversation_id": "A"}],
                          [{"conversation_id": "B"}])

    def test_trainer_rejects_noncopy_type(self):
        a = select_examples([row(1)], self.tok)[0]
        b = select_examples([row(2)], self.tok)[0]
        a["example_type"] = "source_attributed_qa"
        with self.assertRaises(ValueError):
            validate_partitions([a], [b])

    def test_trainer_rejects_modified_target(self):
        a = select_examples([row(1)], self.tok)[0]
        b = select_examples([row(2)], self.tok)[0]
        a["answer"] = "書き換えられた文章"
        with self.assertRaises(ValueError):
            validate_partitions([a], [b])

    def test_trainer_accepts_disjoint_copy(self):
        a = select_examples([row(1)], self.tok)[0]
        b = select_examples([row(2)], self.tok)[0]
        validate_partitions([a], [b])

    def test_trainer_rejects_train_dev_leak(self):
        a = select_examples([row(1)], self.tok)[0]
        b = dict(a)
        b["id"] += ":dev"
        with self.assertRaises(ValueError):
            validate_partitions([a], [b])

    def test_trainer_rejects_empty_dev(self):
        a = select_examples([row(1)], self.tok)[0]
        with self.assertRaises(ValueError):
            validate_partitions([a], [])


if __name__ == "__main__":
    unittest.main()
