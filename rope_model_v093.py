"""LLM_RAG v0.9.3: optional rotary Q/K positions for LLM_GPU v0.3.

Identical parameter names/shapes as the original self-attention. No positional
parameters are learned. RoPE checkpoint configuration is explicitly marked,
and loading it as the legacy model is rejected by the helper below.

This does not add a speaker/turn or larger model.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# LLM_GPU remains an independent local repository; no changes to it.
GPU_DIR = Path(__file__).resolve().parent.parent / "LLM_GPU"
if str(GPU_DIR) not in sys.path:
    sys.path.insert(0, str(GPU_DIR))

from model import LanguageModel, SelfAttention  # noqa: E402


def rotary_transform(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotate paired channels of [batch, seq, channels] by token position."""
    if x.ndim != 3 or x.shape[-1] < 2 or x.shape[-1] % 2:
        raise ValueError("RoPE expects [batch, tokens, even channels >=2]")
    half = x.shape[-1] // 2
    # Pair adjacent coordinates (0,1), (2,3), ...; positions start at 0.
    positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
    inverse = torch.pow(
        torch.tensor(base, device=x.device, dtype=torch.float32),
        -torch.arange(half, device=x.device, dtype=torch.float32) / half,
    )
    angles = positions[:, None] * inverse[None, :]
    cos = torch.cos(angles).to(x.dtype)[None, :, :]
    sin = torch.sin(angles).to(x.dtype)[None, :, :]
    pairs = x.reshape(x.shape[0], x.shape[1], half, 2)
    first, second = pairs.unbind(-1)
    return torch.stack((first * cos - second * sin,
                        first * sin + second * cos), dim=-1).flatten(-2)


class RotarySelfAttention(SelfAttention):
    """Same named parameters and shapes as LLM_GPU SelfAttention."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = rotary_transform(self.q_proj(x))
        k = rotary_transform(self.k_proj(x))
        v = self.v_proj(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(float(self.d_model))
        if self.causal:
            time = x.shape[1]
            mask = torch.triu(
                torch.ones((time, time), device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.out_proj(weights @ v)


class RoPELanguageModel(LanguageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.d_model % 2:
            raise ValueError("RoPE requires an even d_model")
        for block in self.blocks:
            block.attention = RotarySelfAttention(self.d_model, causal=self.causal)

    def config(self):
        value = super().config()
        value["position_encoding"] = "rope_v093"
        return value


def load_configured_model(path, device, *, rope: bool,
                          allow_plain_to_rope: bool = False):
    """Reject implicit reinterpretations of checkpoints across architectures.

    The only allowed architecture change is explicit base->RoPE for INIT
    weights, never a silently reinterpreted RoPE checkpoint.
    """
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no model config")
    marker = config.get("position_encoding", "none")
    if marker not in ("none", "rope_v093"):
        raise ValueError(f"Unsupported position encoding: {marker}")
    if rope and marker == "none" and not allow_plain_to_rope:
        raise ValueError("Explicit --allow-rope-conversion required for plain checkpoint")
    if not rope and marker != "none":
        raise ValueError("RoPE checkpoint cannot be interpreted as a plain model")
    if not rope and allow_plain_to_rope:
        raise ValueError("allow_plain_to_rope requires rope=True")
    klass = RoPELanguageModel if rope else LanguageModel
    accepted = ("vocab_size", "d_model", "num_layers", "hidden_dim",
                "causal", "context_length")
    kwargs = {key: config[key] for key in accepted if key in config}
    model = klass(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    return model, checkpoint
