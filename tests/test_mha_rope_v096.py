"""v0.9.6 MHA correctness and checkpoint compatibility tests."""
import tempfile
import unittest
from pathlib import Path

import torch
from chat import resolve_rope
from mha_rope_v096 import (
    MultiHeadRoPELanguageModel, MultiHeadRotarySelfAttention,
    checkpoint_architecture, load_model_v096, prepare_mha_initial_checkpoint, validate_heads,
)
from rope_model_v093 import RoPELanguageModel


class MultiHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(96)
        self.kw = dict(vocab_size=32, d_model=8, num_layers=2,
                       hidden_dim=16, causal=True, context_length=16)

    def test_invalid_heads(self):
        for h in (0, -1, 3, 8):
            with self.subTest(h=h), self.assertRaises(ValueError):
                validate_heads(8, h)

    def test_head_dim(self):
        self.assertEqual(validate_heads(128, 4), 32)

    def test_parameter_identity(self):
        a = RoPELanguageModel(**self.kw)
        b = MultiHeadRoPELanguageModel(num_heads=2, **self.kw)
        b.load_state_dict(a.state_dict(), strict=True)
        self.assertEqual(a.parameter_count, b.parameter_count)
        for key, v in a.state_dict().items():
            torch.testing.assert_close(v, b.state_dict()[key])

    def test_single_head_equivalence(self):
        a = RoPELanguageModel(**self.kw).eval()
        b = MultiHeadRoPELanguageModel(num_heads=1, **self.kw).eval()
        b.load_state_dict(a.state_dict(), strict=True)
        with torch.no_grad():
            ids = torch.tensor([[1, 2, 3, 4]])
            torch.testing.assert_close(a(ids), b(ids), atol=1e-6, rtol=1e-6)

    def test_causal_mask(self):
        model = MultiHeadRoPELanguageModel(num_heads=2, **self.kw).eval()
        with torch.no_grad():
            a = model(torch.tensor([[1, 2, 3, 4]]))[:, :3]
            b = model(torch.tensor([[1, 2, 3, 7]]))[:, :3]
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_different_heads_change_output(self):
        a = RoPELanguageModel(**self.kw).eval()
        b = MultiHeadRoPELanguageModel(num_heads=2, **self.kw).eval()
        b.load_state_dict(a.state_dict())
        with torch.no_grad():
            ids = torch.tensor([[1, 2, 3, 4]])
            self.assertFalse(torch.allclose(a(ids), b(ids)))

    def test_backward(self):
        m = MultiHeadRoPELanguageModel(num_heads=2, **self.kw)
        m(torch.tensor([[1, 2, 3]])).sum().backward()
        self.assertTrue(all(p.grad is not None for p in m.parameters()))

    def test_shapes(self):
        m = MultiHeadRotarySelfAttention(8, num_heads=2)
        self.assertEqual(tuple(m(torch.randn(3, 5, 8)).shape), (3, 5, 8))

    def test_config(self):
        m = MultiHeadRoPELanguageModel(num_heads=2, **self.kw)
        self.assertEqual(checkpoint_architecture(m.config()), ("mha", 2))

    def test_old_config(self):
        m = RoPELanguageModel(**self.kw)
        self.assertEqual(checkpoint_architecture(m.config()), ("rope", 1))

    def test_bad_marker(self):
        with self.assertRaises(ValueError):
            checkpoint_architecture({"position_encoding": "rope_v093", "num_heads": 4})
        with self.assertRaises(ValueError):
            checkpoint_architecture({"position_encoding": "rope_v096_mha",
                                     "d_model": 8, "num_heads": 1})

    def test_initial_conversion(self):
        m = RoPELanguageModel(**self.kw)
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/"a.pt", Path(tmp)/"b.pt"
            m.save_checkpoint(str(source), epoch=0)
            converted = prepare_mha_initial_checkpoint(source, target, heads=2)
            self.assertEqual(converted.parameter_count, m.parameter_count)
            loaded, _ = load_model_v096(target, torch.device("cpu"),
                                         expected="auto", mha_heads=2)
            self.assertEqual(loaded.num_heads, 2)

    def test_reject_trained_source(self):
        m = RoPELanguageModel(**self.kw)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"trained.pt"
            m.save_checkpoint(str(source), epoch=2, loss=2.0)
            with self.assertRaises(ValueError):
                prepare_mha_initial_checkpoint(source, Path(tmp)/"out.pt", heads=2)

    def test_explicit_conversion_only(self):
        m = RoPELanguageModel(**self.kw)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"init.pt"
            m.save_checkpoint(str(source), epoch=0)
            with self.assertRaises(ValueError):
                load_model_v096(source, torch.device("cpu"), expected="mha", mha_heads=2)
            model, _ = load_model_v096(
                source, torch.device("cpu"), expected="mha", mha_heads=2,
                allow_mha_conversion=True)
            self.assertEqual(model.num_heads, 2)

    def test_head_mismatch_rejected(self):
        m = MultiHeadRoPELanguageModel(num_heads=2, **self.kw)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"mha.pt"
            m.save_checkpoint(str(source), epoch=0)
            with self.assertRaises(ValueError):
                load_model_v096(source, torch.device("cpu"), expected="rope")
            with self.assertRaises(ValueError):
                load_model_v096(source, torch.device("cpu"),
                                expected="mha", mha_heads=4)

    def test_chat_auto_mha(self):
        cfg = {"position_encoding": "rope_v096_mha", "num_heads": 4}
        self.assertTrue(resolve_rope("auto", cfg))
        self.assertTrue(resolve_rope("mha", cfg))
        with self.assertRaises(ValueError):
            resolve_rope("rope", cfg)


if __name__ == "__main__":
    unittest.main()
