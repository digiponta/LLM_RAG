"""LLM_RAG v0.9.6: multi-head rotary attention with checkpoint-safe opt-in.

Each head applies RoPE independently to its own (d_model / num_heads)
Q/K coordinates; head scores scale by sqrt(head_dim). This preserves ALL
learnable parameter names/shapes of single-head RoPE. No code changes to
the sibling LLM_GPU repo. Existing single-head checkpoints are never
silently treated as multi-head.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

GPU_DIR = Path(__file__).resolve().parent.parent / "LLM_GPU"
if str(GPU_DIR) not in sys.path:
    sys.path.insert(0, str(GPU_DIR))

from model import SelfAttention
from rope_model_v093 import RoPELanguageModel, rotary_transform, load_configured_model

POSITION_ENCODING_MHA = "rope_v096_mha"


def validate_heads(d_model: int, num_heads: int) -> int:
    if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads < 1:
        raise ValueError("num_heads must be a positive integer")
    if d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    head_dim = d_model // num_heads
    if head_dim < 2 or head_dim % 2:
        raise ValueError("RoPE requires an even head_dim >=2")
    return head_dim


class MultiHeadRotarySelfAttention(SelfAttention):
    """Parameter-identical multi-head reinterpretation of v0.9.3 attention."""

    def __init__(self, d_model: int, num_heads: int = 4, causal: bool = True):
        validate_heads(d_model, num_heads)
        super().__init__(d_model, causal=causal)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def split(self, value: torch.Tensor) -> torch.Tensor:
        batch, time, _ = value.shape
        return value.reshape(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        q = self.split(self.q_proj(x))
        k = self.split(self.k_proj(x))
        v = self.split(self.v_proj(x))
        # rotary_transform rotates the last-dim pairs and expects 3D,
        # so fold [B,H] into one axis WITHOUT mixing heads.
        q = rotary_transform(q.reshape(batch * self.num_heads, time, self.head_dim))
        k = rotary_transform(k.reshape(batch * self.num_heads, time, self.head_dim))
        q = q.reshape(batch, self.num_heads, time, self.head_dim)
        k = k.reshape(batch, self.num_heads, time, self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        if self.causal:
            causal_mask = torch.ones((time, time), dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        values = attention @ v
        merged = values.transpose(1, 2).contiguous().reshape(batch, time, self.d_model)
        return self.out_proj(merged)


class MultiHeadRoPELanguageModel(RoPELanguageModel):
    """RoPE LM with preserved Q/K/V projections; num_heads is fixed in config."""

    def __init__(self, *, num_heads: int = 4, **kwargs):
        d_model = kwargs.get("d_model", 64)
        validate_heads(d_model, num_heads)
        super().__init__(**kwargs)
        self.num_heads = num_heads
        for block in self.blocks:
            block.attention = MultiHeadRotarySelfAttention(
                self.d_model, num_heads=num_heads, causal=self.causal
            )

    def config(self):
        cfg = super().config()
        cfg["position_encoding"] = POSITION_ENCODING_MHA
        cfg["num_heads"] = self.num_heads
        return cfg


def checkpoint_architecture(config: dict) -> tuple[str, int]:
    """Marker and head count are trusted together. Legacy configs lack marker."""
    marker = config.get("position_encoding", "none")
    if marker == "none":
        if config.get("num_heads", 1) != 1:
            raise ValueError("Legacy checkpoint must have one head")
        return "legacy", 1
    if marker == "rope_v093":
        if config.get("num_heads", 1) != 1:
            raise ValueError("Single-head RoPE checkpoint has inconsistent heads")
        return "rope", 1
    if marker == POSITION_ENCODING_MHA:
        heads = config.get("num_heads")
        validate_heads(int(config["d_model"]), heads)
        if heads < 2:
            raise ValueError("Multi-head checkpoint must contain >=2 heads")
        return "mha", heads
    raise ValueError(f"Unsupported position_encoding marker: {marker}")


def load_model_v096(path: Path, device: torch.device, *, expected: str = "auto",
                    mha_heads: int = 4, allow_mha_conversion: bool = False,
                    allow_rope_conversion: bool = False):
    """Loads exact marked architecture, or explicitly converts a fresh init.

    Single-head RoPE to MHA reuses all weights but changes attention outputs;
    never call this an inference-equivalent checkpoint conversion. Conversion
    is permitted ONLY when explicitly requested, and is intended for fresh
    initial checkpoints (training must follow).
    """
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint missing config")
    actual, heads = checkpoint_architecture(config)
    if expected not in ("auto", "legacy", "rope", "mha"):
        raise ValueError("expected must be auto|legacy|rope|mha")
    requested = actual if expected == "auto" else expected
    if requested == "mha":
        if actual == "mha":
            if mha_heads != heads:
                raise ValueError("MHA checkpoint head count mismatch")
        elif actual == "rope" and allow_mha_conversion:
            validate_heads(int(config["d_model"]), mha_heads)
            if mha_heads < 2:
                raise ValueError("MHA requires >=2 heads")
        else:
            raise ValueError("Single-head -> MHA conversion requires explicit opt-in and a RoPE checkpoint")
        ctor_keys = ("vocab_size", "d_model", "num_layers",
                     "hidden_dim", "causal", "context_length")
        cfg = {key: config[key] for key in ctor_keys if key in config}
        model = MultiHeadRoPELanguageModel(num_heads=mha_heads, **cfg)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        return model, checkpoint
    if requested != actual:
        if requested == "rope" and actual == "legacy" and allow_rope_conversion:
            return load_configured_model(path, device, rope=True, allow_plain_to_rope=True)
        raise ValueError(f"Checkpoint architecture {actual!r} != requested {requested!r}")
    if actual == "mha":
        raise AssertionError("Internal MHA loader dispatch failure")
    return load_configured_model(path, device, rope=(actual == "rope"))


def prepare_mha_initial_checkpoint(source: Path, output: Path, *, heads: int):
    """Convert a verified *fresh* single-head RoPE init; refuse trained models."""
    if output.resolve() == source.resolve():
        raise ValueError("Source and output checkpoints must differ")
    checkpoint = torch.load(str(source), map_location="cpu", weights_only=True)
    architecture, _ = checkpoint_architecture(checkpoint["config"])
    if architecture != "rope":
        raise ValueError("MHA init conversion requires a single-head RoPE checkpoint")
    if checkpoint.get("epoch") != 0:
        raise ValueError("Only an explicitly fresh epoch=0 checkpoint is supported")
    model, _ = load_model_v096(source, torch.device("cpu"),
                              expected="mha", mha_heads=heads,
                              allow_mha_conversion=True)
    # initialize no new learnable parameters: exactly same state_dict values.
    model.save_checkpoint(str(output), epoch=0, loss=None)
    return model
