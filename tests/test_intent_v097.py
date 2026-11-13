"""v0.9.7 tests for weak-intent auxiliaries and disjoint concept examples."""
import tempfile
import unittest
from pathlib import Path

import torch

from intent_model_v097 import (
    INTENTS, UNKNOWN_INTENT, IntentRoPELanguageModel,
    initialize_intent_model, load_intent_model, weak_intent,
)
from prepare_concepts_v097 import CONCEPTS, build_rows, run as prepare_run
from train_intent_v097 import collate_intent, prepare_labeled, training_losses


class StubTokenizer:
    pad_id = 0
    eos_id = 2
    unk_id = 3
    def encode(self, text, add_bos=False):
        tokens = [10 + (ord(c) % 20) for c in text]
        return ([1] if add_bos else []) + tokens


class IntentV097Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)
        self.kw = dict(vocab_size=32, d_model=8, num_layers=2,
                       hidden_dim=16, causal=True, context_length=64)

    def test_definition_intent(self):
        for question in ("宇宙とは", "時間とは", "空間とは、", "光とは何ですか？"):
            with self.subTest(question=question):
                self.assertEqual(INTENTS[weak_intent(question)], "DEFINITION")

    def test_greeting_intent(self):
        self.assertEqual(INTENTS[weak_intent("こんにちは")], "GREETING")

    def test_fact_intent(self):
        self.assertEqual(INTENTS[weak_intent("化学組成は何ですか？")], "FACT_QUESTION")

    def test_preference_intent(self):
        self.assertEqual(INTENTS[weak_intent("読書は好きですか？")], "PREFERENCE")

    def test_feeling_intent(self):
        self.assertEqual(INTENTS[weak_intent("今日は疲れました")], "FEELING")

    def test_request_intent(self):
        self.assertEqual(INTENTS[weak_intent("説明してください")], "REQUEST")

    def test_ambiguous_masked(self):
        self.assertEqual(weak_intent("貴方は"), UNKNOWN_INTENT)

    def test_empty_masked(self):
        self.assertEqual(weak_intent("  "), UNKNOWN_INTENT)

    def test_forward_shapes(self):
        model = IntentRoPELanguageModel(**self.kw)
        x = torch.tensor([[1, 2, 4, 5], [1, 4, 5, 6]])
        lm, auxiliary = model.forward_with_intent(x, torch.tensor([1, 2]))
        self.assertEqual(tuple(lm.shape), (2, 4, 32))
        self.assertEqual(tuple(auxiliary.shape), (2, len(INTENTS)))

    def test_intent_does_not_see_future_answer(self):
        model = IntentRoPELanguageModel(**self.kw).eval()
        with torch.no_grad():
            _, a = model.forward_with_intent(torch.tensor([[1, 2, 3, 4]]), torch.tensor([1]))
            _, b = model.forward_with_intent(torch.tensor([[1, 2, 9, 9]]), torch.tensor([1]))
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

    def test_position_validation(self):
        model = IntentRoPELanguageModel(**self.kw)
        with self.assertRaises(ValueError):
            model.forward_with_intent(torch.tensor([[1, 2]]), torch.tensor([2]))

    def test_loss_zero_intent_weight(self):
        model = IntentRoPELanguageModel(**self.kw)
        x = torch.tensor([[1, 2, 3, 4]])
        y = torch.tensor([[-100, 2, 3, 4]])
        total, language, intent = training_losses(
            model, x, y, torch.tensor([1]), torch.tensor([weak_intent("宇宙とは")]), 0.0)
        torch.testing.assert_close(total, language)

    def test_loss_unknown_only_finite(self):
        model = IntentRoPELanguageModel(**self.kw)
        x = torch.tensor([[1, 2, 3, 4]])
        y = torch.tensor([[-100, 2, 3, 4]])
        total, language, intent = training_losses(
            model, x, y, torch.tensor([1]), torch.tensor([UNKNOWN_INTENT]), .2)
        self.assertTrue(torch.isfinite(total).item())
        self.assertEqual(intent.item(), 0.)

    def test_gradients_reach_intent_head(self):
        model = IntentRoPELanguageModel(**self.kw)
        x = torch.tensor([[1, 2, 3, 4]])
        y = torch.tensor([[-100, 2, 3, 4]])
        total, _, _ = training_losses(
            model, x, y, torch.tensor([1]), torch.tensor([weak_intent("宇宙とは")]), .2)
        total.backward()
        self.assertIsNotNone(model.intent_head.weight.grad)

    def test_collate_position_correct(self):
        items = [
            (((1, 2, 3, 4), (-100, -100, 3, 4)), 1),
            (((1, 2, 3), (-100, 2, 3)), UNKNOWN_INTENT),
        ]
        ids, labels, position, intents = collate_intent(items, 0)
        self.assertEqual(position.tolist(), [2, 1])
        self.assertEqual(intents.tolist(), [1, -100])

    def test_checkpoint_transfer_preserves_backbone(self):
        from rope_model_v093 import RoPELanguageModel
        base = RoPELanguageModel(**self.kw)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"source.pt"
            out = Path(tmp)/"intent.pt"
            base.save_checkpoint(str(source), epoch=2)
            result, _ = initialize_intent_model(source, device="cpu")
            self.assertEqual(result.config()["auxiliary_head"], "dialogue_intent_v097")
            for name, value in base.state_dict().items():
                torch.testing.assert_close(value, result.state_dict()[name])
            result.save_checkpoint(str(out), epoch=1)
            loaded, _ = load_intent_model(out, torch.device("cpu"))
            self.assertEqual(loaded.parameter_count, result.parameter_count)

    def test_reject_wrong_intent_head_marker(self):
        from rope_model_v093 import RoPELanguageModel
        base = RoPELanguageModel(**self.kw)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"base.pt"
            base.save_checkpoint(str(path), epoch=1)
            with self.assertRaises(ValueError):
                load_intent_model(path, torch.device("cpu"))

    def test_concept_splits_disjoint(self):
        names = set()
        for split in CONCEPTS:
            for term, _ in CONCEPTS[split]:
                self.assertNotIn(term, names)
                names.add(term)

    def test_concept_examples_have_local_ids(self):
        self.assertEqual(build_rows("test")[0]["question"], "宇宙とは")
        self.assertEqual(build_rows("train")[0]["example_type"], "hand_authored_definition_probe")

    def test_concept_preparation_non_overwrite(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output_dir=Path(tmp), overwrite=False)
            paths = prepare_run(args)
            self.assertTrue(paths["train"].exists())
            with self.assertRaises(FileExistsError):
                prepare_run(args)


if __name__ == "__main__":
    unittest.main()
