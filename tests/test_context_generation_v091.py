import unittest

from evaluate_context_generation_v091 import (
    plain_variant, display_case, summarize_cases, case_hash,
)


class ContextGenerationV091Tests(unittest.TestCase):
    def test_plain_variant_preserves_answer(self):
        original = {"id": "a:1", "context": "過去の話", "question": "あなたは？",
                    "answer": "読書です", "conversation_id": "a"}
        converted = plain_variant(original)
        self.assertEqual(converted["context"], "")
        self.assertEqual(converted["question"], original["question"])
        self.assertEqual(converted["answer"], original["answer"])
        self.assertEqual(original["context"], "過去の話")

    def test_hash_private(self):
        self.assertEqual(case_hash("秘密"), case_hash("秘密"))
        self.assertEqual(len(case_hash("秘密")), 16)
        self.assertNotIn("秘密", case_hash("秘密"))

    def test_summary_zero(self):
        self.assertEqual(summarize_cases([]), {"cases": 0})

    def test_summary_all_skipped(self):
        self.assertEqual(summarize_cases([{"status": "SKIPPED"}]),
                         {"cases": 1, "valid": 0})

    def test_summary_valid(self):
        rows = [
            {"status": "OK", "eos": True, "exact_match": True,
             "chars": 3, "response_hash": "abc"},
            {"status": "OK", "eos": False, "exact_match": False,
             "chars": 5, "response_hash": "abc"},
        ]
        result = summarize_cases(rows)
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["valid"], 2)
        self.assertEqual(result["unique_response_ratio"], .5)
        self.assertEqual(result["eos_rate"], .5)
        self.assertEqual(result["exact_match_rate"], .5)
        self.assertEqual(result["mean_chars"], 4)

    def test_summary_ignores_skipped(self):
        rows = [
            {"status": "SKIPPED"},
            {"status": "OK", "eos": True, "exact_match": False,
             "chars": 2, "response_hash": "1"},
        ]
        result = summarize_cases(rows)
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["valid"], 1)

    def test_review_history_order_and_labels(self):
        row = {"context": "寒いですね／厚着しています", "question": "あなたは？",
               "answer": "もこもこ着ています", "question_speaker": "AA",
               "answer_speaker": "AB"}
        result = {"context_generation": {"text": "そうですね"},
                  "plain_generation": {"text": "はい"},
                  "context_answer_nll": 2.0, "plain_answer_nll": 2.2,
                  "context_nll_advantage": .2, "response_changed": True}
        review = display_case(row, result)
        self.assertEqual(review["history"], ["寒いですね", "厚着しています"])
        self.assertEqual(review["question_speaker"], "AA")
        self.assertEqual(review["answer_speaker"], "AB")
        self.assertTrue(review["response_changed"])

    def test_review_no_context(self):
        row = {"context": "", "question": "こんにちは", "answer": "はい"}
        result = {"context_generation": {"text": "はい"},
                  "plain_generation": {"text": "はい"},
                  "context_answer_nll": 1.0, "plain_answer_nll": 1.0,
                  "context_nll_advantage": 0.0, "response_changed": False}
        self.assertEqual(display_case(row, result)["history"], [])


if __name__ == "__main__":
    unittest.main()
