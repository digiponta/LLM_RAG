"""Architecture and numerical checks for v0.9.3 RoPE optional attention."""
import tempfile
import unittest
from pathlib import Path

import torch

from rope_model_v093 import (
    LanguageModel, RoPELanguageModel, rotary_transform, load_configured_model,
)


class RoPEV093Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(93)
        self.kwargs = dict(vocab_size=32, d_model=8, num_layers=2,
                           hidden_dim=16, context_length=16, causal=True)

    def test_position_zero_identity(self):
        x = torch.randn(2, 3, 8)
        rotated = rotary_transform(x)
        torch.testing.assert_close(rotated[:, 0], x[:, 0], atol=1e-6, rtol=1e-6)

    def test_rotation_changes_later_position(self):
        x = torch.ones(1, 3, 8)
        rotated = rotary_transform(x)
        self.assertFalse(torch.allclose(rotated[:, 1], x[:, 1]))

    def test_rotation_preserves_norm(self):
        x = torch.randn(2, 7, 8)
        rotated = rotary_transform(x)
        torch.testing.assert_close(torch.linalg.norm(rotated, dim=-1),
                                   torch.linalg.norm(x, dim=-1), atol=1e-5, rtol=1e-5)

    def test_odd_dim_rejected(self):
        with self.assertRaises(ValueError):
            rotary_transform(torch.randn(1, 4, 7))

    def test_named_weights_and_parameter_count_same(self):
        plain = LanguageModel(**self.kwargs)
        rope = RoPELanguageModel(**self.kwargs)
        self.assertEqual(plain.parameter_count, rope.parameter_count)
        self.assertEqual({k: v.shape for k, v in plain.state_dict().items()},
                         {k: v.shape for k, v in rope.state_dict().items()})

    def test_causal_future_cannot_change_earlier_logits(self):
        model = RoPELanguageModel(**self.kwargs).eval()
        a = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        b = torch.tensor([[1, 2, 3, 9]], dtype=torch.long)
        with torch.no_grad():
            first = model(a)[:, :3]
            second = model(b)[:, :3]
        torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)

    def test_backprop_works(self):
        model = RoPELanguageModel(**self.kwargs)
        output = model(torch.tensor([[1, 2, 3, 4]]))
        output.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_plain_to_rope_requires_explicit_conversion(self):
        plain = LanguageModel(**self.kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "init.pt"
            plain.save_checkpoint(str(path))
            with self.assertRaises(ValueError):
                load_configured_model(path, torch.device("cpu"), rope=True)
            converted, _ = load_configured_model(
                path, torch.device("cpu"), rope=True, allow_plain_to_rope=True)
            self.assertIsInstance(converted, RoPELanguageModel)
            for key, value in plain.state_dict().items():
                torch.testing.assert_close(converted.state_dict()[key], value)

    def test_rope_checkpoint_plain_load_refused(self):
        rope = RoPELanguageModel(**self.kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rope.pt"
            rope.save_checkpoint(str(path))
            with self.assertRaises(ValueError):
                load_configured_model(path, torch.device("cpu"), rope=False)
            restored, _ = load_configured_model(path, torch.device("cpu"), rope=True)
            self.assertEqual(restored.config()["position_encoding"], "rope_v093")

    def test_plain_checkpoint_plain_loader(self):
        plain = LanguageModel(**self.kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.pt"
            plain.save_checkpoint(str(path))
            model, _ = load_configured_model(path, torch.device("cpu"), rope=False)
            self.assertIsInstance(model, LanguageModel)
            self.assertNotIsInstance(model, RoPELanguageModel)

    def test_no_odd_model_dimensions(self):
        kwargs = dict(self.kwargs, d_model=7)
        with self.assertRaises(ValueError):
            RoPELanguageModel(**kwargs)

    def test_rope_config_saved(self):
        model = RoPELanguageModel(**self.kwargs)
        self.assertEqual(model.config()["position_encoding"], "rope_v093")


if __name__ == "__main__":
    unittest.main()
