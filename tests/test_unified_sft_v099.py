"""v0.9.9 unified SFT checks; all tests use synthetic corpus, never private data."""
import tempfile
import unittest
from pathlib import Path

from prepare_real_persona_v081 import compact_prompt
from prepare_unified_sft_v099 import (
    check_splits, make_nagato_rows, mix_training_rows,
    segment_text, split_for_line, write_jsonl,
)
from train_real_dialogue_v082 import encode_dialogue, load_dialogue_rows


class CharacterTokenizerStub:
    eos_id = 2
    unk_id = 3
    def encode(self, text, add_bos=False):
        return ([1] if add_bos else []) + [10 + (ord(c) % 30) for c in text]


class UnifiedV099Tests(unittest.TestCase):
    def sample_corpus(self):
        return "@NagatFakeBot heading\n" + "\n".join(
            f"宇宙についての考え方その{i}。時間や空間の観測についての記述。"
            for i in range(80)
        )

    def test_nagato_rows_are_real_sft_schema(self):
        rows = make_nagato_rows(self.sample_corpus())
        example = next(x for x in rows["train"] if x["example_type"] == "source_attributed_qa")
        self.assertTrue(all(isinstance(example[k], str) and example[k]
                            for k in ("id", "conversation_id", "question", "answer")))
        self.assertEqual(example["context"], "")
        self.assertIn("長門コーパスには「", example["answer"])
        self.assertIn("」と記されています。", example["answer"])
        self.assertIn("問:", compact_prompt(example))
        self.assertIn("答:", compact_prompt(example))

    def test_source_and_persona_variants_share_schema_and_origin(self):
        splits = make_nagato_rows("宇宙に関する思索です。")
        variants = [row for rows in splits.values() for row in rows]
        self.assertEqual(len(variants), 2)
        self.assertEqual({v["example_type"] for v in variants},
                         {"source_attributed_qa", "persona_style_from_source"})
        self.assertEqual(len({v["conversation_id"] for v in variants}), 1)
        self.assertEqual(len({v["id"] for v in variants}), 2)

    def test_split_stable(self):
        self.assertEqual(split_for_line("宇宙とは"), split_for_line("宇宙とは"))

    def test_split_reject_bad_fractions(self):
        with self.assertRaises(ValueError):
            split_for_line("宇宙とは", validation_fraction=.9, test_fraction=.2)

    def test_no_original_line_cross_split(self):
        split_rows = make_nagato_rows(self.sample_corpus(), max_chars=24)
        check_splits(split_rows)
        locations = {}
        for split, rows in split_rows.items():
            for row in rows:
                origin = row["original_line_hash"]
                self.assertEqual(locations.setdefault(origin, split), split)

    def test_duplicate_lines_deduplicated(self):
        source = "\n".join(["宇宙は広がっています。"] * 10)
        split_rows = make_nagato_rows(source)
        self.assertEqual(sum(len(x) for x in split_rows.values()), 2)

    def test_long_lines_chunked_without_fabrication(self):
        line = "宇宙の観測。" * 40
        chunks = segment_text(line, max_chars=30)
        self.assertEqual("".join(chunks), line)
        self.assertTrue(all(len(x) <= 30 for x in chunks))

    def test_header_omitted(self):
        rows = make_nagato_rows("@NagatFakeBot heading")
        self.assertEqual(sum(len(x) for x in rows.values()), 0)

    def test_encoded_by_shared_trainer(self):
        rows = make_nagato_rows(self.sample_corpus())
        row = next(iter(rows["train"]))
        encoded, status = encode_dialogue(row, CharacterTokenizerStub(), 256)
        self.assertEqual(status, "accepted")
        self.assertEqual(len(encoded[0]), len(encoded[1]))

    def test_train_sampling_deterministic(self):
        nagato = [{"id": "n1", "conversation_id": "n1",
                   "question": "A", "answer": "B", "context": ""}]
        persona = [{"id": f"p{i}", "conversation_id": f"p{i}",
                    "question": "A", "answer": "B", "context": ""}
                   for i in range(19)]
        a = mix_training_rows(persona, nagato, nagato_fraction=.1, seed=9)
        b = mix_training_rows(persona, nagato, nagato_fraction=.1, seed=9)
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])
        self.assertEqual(sum(r["source_kind"] == "nagato_corpus" for r in a), 2)

    def test_distinct_repeat_ids_same_provenance(self):
        nagato = [{"id": "n1", "conversation_id": "nagato-line",
                   "question": "Q", "answer": "A", "context": ""}]
        persona = [{"id": f"p{i}", "conversation_id": f"persona{i}",
                    "question": "Q", "answer": "A", "context": ""}
                   for i in range(100)]
        mixed = mix_training_rows(persona, nagato, nagato_fraction=.1)
        sampled = [r for r in mixed if r["source_kind"] == "nagato_corpus"]
        self.assertEqual(len(sampled), len({r["id"] for r in sampled}))
        self.assertEqual({r["conversation_id"] for r in sampled}, {"nagato-line"})

    def test_ratio_invalid(self):
        with self.assertRaises(ValueError):
            mix_training_rows([{"id": "a"}], [{"id": "b"}], nagato_fraction=.5)

    def test_empty_source_rejected(self):
        with self.assertRaises(ValueError):
            mix_training_rows([], [{"id": "b"}])

    def test_cross_split_leak_detected(self):
        row = {"id": "id1", "conversation_id": "one"}
        with self.assertRaises(ValueError):
            check_splits({"train": [row], "validation": [{**row, "id": "id2"}]})

    def test_serialize_and_train_loader(self):
        row = {"id": "test", "conversation_id": "test-source",
               "question": "宇宙とは", "answer": "資料では述べています。", "context": ""}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            write_jsonl(path, [row])
            loaded = load_dialogue_rows(path, 0, 42)
            self.assertEqual(loaded, [row])

    def test_raw_source_statement_preserved(self):
        statement = "量子についての意見です。"
        rows = make_nagato_rows(statement)
        values = [r["answer"] for items in rows.values() for r in items]
        self.assertEqual(set(values), {
            "長門コーパスには「" + statement + "」と記されています。",
            statement,
        })


if __name__ == "__main__":
    unittest.main()
