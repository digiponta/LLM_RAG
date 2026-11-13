import unittest

from prepare_speaker_turn_v092 import structured_prompt, reconstruct
from prepare_real_persona_v081 import compact_prompt


class SpeakerTurnV092Tests(unittest.TestCase):
    def setUp(self):
        self.dialogue = {
            "utterances": [
                {"interlocutor_id": "AB", "text": "寒いですね"},
                {"interlocutor_id": "AA", "text": "暖かい服を着ます"},
                {"interlocutor_id": "AB", "text": "あなたは？"},
                {"interlocutor_id": "AA", "text": "厚着です"},
            ]
        }
        self.base = {
            "id": "dialogue.json:3", "conversation_id": "dialogue.json",
            "question": "あなたは？", "answer": "厚着です",
            "question_speaker": "AB", "answer_speaker": "AA",
            "context": "寒いですね／暖かい服を着ます",
        }

    def test_structured_prompt_order(self):
        prompt = structured_prompt("あなたは？", "AB", "AA",
                                  [("AB", "寒いですね"), ("AA", "暖かい服を着ます")])
        self.assertEqual(prompt, "B:寒いですね\nA:暖かい服を着ます\nB:あなたは？\nA:")

    def test_reconstruct(self):
        output = reconstruct(self.base, self.dialogue, 2)
        self.assertEqual(output["prompt_override"],
                         "B:寒いですね\nA:暖かい服を着ます\nB:あなたは？\nA:")
        self.assertEqual(output["question"], self.base["question"])
        self.assertEqual(output["answer"], self.base["answer"])

    def test_legacy_unmodified(self):
        self.assertEqual(compact_prompt(self.base),
                         "前:寒いですね／暖かい服を着ます\n問:あなたは？\n答:")

    def test_override_used(self):
        output = reconstruct(self.base, self.dialogue, 2)
        self.assertEqual(compact_prompt(output), output["prompt_override"])

    def test_original_row_not_mutated(self):
        reconstruct(self.base, self.dialogue, 2)
        self.assertNotIn("prompt_override", self.base)

    def test_reject_wrong_answer(self):
        row = dict(self.base, answer="違う回答")
        with self.assertRaises(ValueError):
            reconstruct(row, self.dialogue, 2)

    def test_reject_changed_source_context(self):
        row = dict(self.base, context="別の文脈")
        with self.assertRaises(ValueError):
            reconstruct(row, self.dialogue, 2)

    def test_reject_wrong_speaker(self):
        row = dict(self.base, question_speaker="AA")
        with self.assertRaises(ValueError):
            reconstruct(row, self.dialogue, 2)

    def test_reject_other_speaker(self):
        with self.assertRaises(ValueError):
            structured_prompt("質問", "AB", "AA", [("CC", "別話者")])

    def test_invalid_roles(self):
        with self.assertRaises(ValueError):
            structured_prompt("質問", "AB", "AB", [])

    def test_invalid_prompt_override(self):
        with self.assertRaises(ValueError):
            compact_prompt({"prompt_override": "B:質問\n", "question": "質問"})

    def test_context_one_turn(self):
        row = dict(self.base, context="暖かい服を着ます")
        value = reconstruct(row, self.dialogue, 1)["prompt_override"]
        self.assertEqual(value, "A:暖かい服を着ます\nB:あなたは？\nA:")


if __name__ == "__main__":
    unittest.main()
