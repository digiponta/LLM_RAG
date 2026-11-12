import random
import unittest
from evaluate_context_swap_v088 import choose_donor, short_hash, summarize


class TokenizerStub:
    unk_id = -1
    eos_id = 2
    def encode(self, text, add_bos=False):
        return ([1] if add_bos else []) + [ord(c) for c in text]


def row(ident, conversation, context, question="あなたは？"):
    return {"id": ident, "conversation_id": conversation, "context": context,
            "question": question, "answer": "厚着です"}


class ContextSwapV088Tests(unittest.TestCase):
    def setUp(self):
        self.tok = TokenizerStub()
        self.target = row("a:1", "a", "冬の服", "あなたは？")
        self.same_chat = row("a:2", "a", "違う話", "元気？")
        self.other = row("b:1", "b", "趣味の話", "趣味は？")

    def test_no_same_dialogue_donor(self):
        selected = choose_donor(self.target, [self.same_chat, self.other], self.tok, 64,
                                random.Random(1), "random")
        self.assertEqual(selected["donor"]["id"], "b:1")

    def test_no_same_question_donor(self):
        candidate = row("b:1", "b", "趣味の話", "あなたは？")
        self.assertIsNone(choose_donor(self.target, [candidate], self.tok, 64,
                                       random.Random(1), "random"))

    def test_no_same_context_donor(self):
        candidate = row("b:1", "b", "冬の服", "趣味は？")
        self.assertIsNone(choose_donor(self.target, [candidate], self.tok, 64,
                                       random.Random(1), "random"))

    def test_no_self_donor(self):
        self.assertIsNone(choose_donor(self.target, [self.target], self.tok, 64,
                                       random.Random(1), "random"))

    def test_length_matched(self):
        near = row("c:1", "c", "春の服", "趣味は？")
        far = row("b:1", "b", "とてもとても長い趣味の話", "趣味は？")
        selected = choose_donor(self.target, [far, near], self.tok, 64,
                                random.Random(1), "length_matched", 0)
        self.assertEqual(selected["donor"]["id"], "c:1")
        self.assertEqual(selected["length_delta"], 0)

    def test_no_eligible_length(self):
        far = row("b:1", "b", "長い文脈があるのです", "趣味は？")
        self.assertIsNone(choose_donor(self.target, [far], self.tok, 64,
                                       random.Random(1), "length_matched", 0))

    def test_random_reproducible(self):
        donors = [row(f"{i}:1", str(i), f"別の話{i}", f"質問{i}") for i in range(5)]
        a = choose_donor(self.target, donors, self.tok, 64, random.Random(17), "random")
        b = choose_donor(self.target, donors, self.tok, 64, random.Random(17), "random")
        self.assertEqual(a["donor"]["id"], b["donor"]["id"])

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            choose_donor(self.target, [], self.tok, 64, random.Random(0), "broken")

    def test_summary_empty(self):
        self.assertEqual(summarize([]), {"cases": 0})

    def test_summary_margins(self):
        cases = [
            {"margin": .4, "correct_nll": 2., "swapped_nll": 2.4, "length_delta": 0},
            {"margin": -.2, "correct_nll": 2., "swapped_nll": 1.8, "length_delta": 2},
        ]
        result = summarize(cases)
        self.assertEqual(result["cases"], 2)
        self.assertAlmostEqual(result["mean_nll_margin"], .1)
        self.assertEqual(result["positive_margin_rate"], .5)
        self.assertAlmostEqual(result["mean_context_token_length_delta"], 1)

    def test_private_hash(self):
        value = short_hash("private content")
        self.assertEqual(value, short_hash("private content"))
        self.assertEqual(len(value), 16)
        self.assertNotIn("private content", value)

    def test_swapped_answer_and_question_unchanged(self):
        chosen = choose_donor(self.target, [self.other], self.tok, 64,
                              random.Random(1), "random")
        self.assertEqual(chosen["swapped"]["answer"], self.target["answer"])
        self.assertEqual(chosen["swapped"]["question"], self.target["question"])
        self.assertNotEqual(chosen["swapped"]["context"], self.target["context"])


if __name__ == "__main__":
    unittest.main()
