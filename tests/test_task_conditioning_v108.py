"""v0.10.8 CPU regression: tagged prompts preserve answer labels and source splits."""
import unittest

from task_conditioning_v108 import TAG_BY_TASK, tagged
from prepare_task_conditioned_v108 import transform
from prepare_real_persona_v081 import compact_prompt
from train_real_dialogue_v082 import encode_dialogue


class FakeTokenizer:
    unk_id = 1
    eos_id = 3
    def encode(self, text, add_bos=False):
        return ([2] if add_bos else []) + [ord(ch) + 10 for ch in text]


class V108Tests(unittest.TestCase):
    def make(self, task="position"):
        return {"id": "sample:1", "conversation_id": "dialogue:1",
                "question": "5文字目は？", "context": "資料:ABCDEFGHIJ",
                "answer": "E", "multitask": task}

    def test_six_distinct_markers(self):
        self.assertEqual(len(TAG_BY_TASK), 6)
        self.assertEqual(len(set(TAG_BY_TASK.values())), 6)

    def test_expected_position_marker(self):
        self.assertEqual(tagged(self.make())["task_marker"], "[TASK:POSITION]")

    def test_all_tasks_supported(self):
        for task in TAG_BY_TASK:
            self.assertTrue(compact_prompt(tagged(self.make(task))).startswith("前:[TASK:"))

    def test_knowledge_marker(self):
        self.assertIn("[TASK:KNOWLEDGE]", compact_prompt(tagged(self.make("nagato_knowledge"))))

    def test_context_preserved(self):
        self.assertEqual(tagged(self.make())["context"], "[TASK:POSITION]\n資料:ABCDEFGHIJ")

    def test_no_context_still_tagged(self):
        row = self.make("real_persona")
        row["context"] = ""
        self.assertEqual(tagged(row)["context"], "[TASK:CHAT]")

    def test_answer_unchanged(self):
        row = self.make()
        self.assertEqual(tagged(row)["answer"], row["answer"])

    def test_source_unmodified(self):
        row = self.make()
        tagged(row)
        self.assertEqual(row["context"], "資料:ABCDEFGHIJ")

    def test_reject_double_tag(self):
        row = self.make()
        with self.assertRaises(ValueError):
            tagged(tagged(row))

    def test_reject_unknown_task(self):
        with self.assertRaises(ValueError):
            tagged(self.make(), "unrecognized")

    def test_reject_override(self):
        row = self.make()
        row["prompt_override"] = "Q:foo\nA:"
        with self.assertRaises(ValueError):
            tagged(row)

    def test_encode_answer_eos(self):
        tok = FakeTokenizer()
        pair, status = encode_dialogue(tagged(self.make()), tok, 256,
                                        max_unknown_fraction=.1, min_answer_tokens=1)
        self.assertEqual(status, "accepted")
        self.assertEqual(pair[1][-1], tok.eos_id)

    def test_train_and_inference_prompt_identical(self):
        row = self.make()
        train = tagged(row)
        inference = tagged(row, "position")
        self.assertEqual(compact_prompt(train), compact_prompt(inference))

    def test_filter_mixed_with_one_token(self):
        tr, stats = transform([self.make()], FakeTokenizer(), 256)
        self.assertEqual(len(tr), 1)
        self.assertEqual(stats["position:accepted"], 1)

    def test_tag_does_not_affect_conversation_id(self):
        row = self.make()
        self.assertEqual(tagged(row)["conversation_id"], row["conversation_id"])


if __name__ == "__main__":
    unittest.main()
