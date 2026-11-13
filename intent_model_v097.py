"""LLM_RAG v0.9.7: optional dialogue-intent auxiliary classification head.

This is an experimental auxiliary loss, not an answer router or semantic gate.
The base autoregressive forward/generate interface remains intact.
"""
from __future__ import annotations

import re
import torch
from torch import nn
from rope_model_v093 import RoPELanguageModel

INTENTS = (
    "GREETING", "DEFINITION", "FACT_QUESTION", "PREFERENCE",
    "FEELING", "REQUEST", "OTHER",
)
UNKNOWN_INTENT = -100


def weak_intent(question: str) -> int:
    """High-precision heuristic pseudo-labels; exclude ambiguous utterances.

    Deliberately do NOT call these human-verified semantic labels. OTHER is
    reserved for future human-reviewed examples; unknown examples are masked.
    """
    q = re.sub(r"\s+", "", question or "").strip("。！？!?、, ")
    if not q:
        return UNKNOWN_INTENT
    if re.fullmatch(r"(こんにちは|こんばんは|おはよう(?:ございます)?|はじめまして|よろしくお願いします)", q):
        return INTENTS.index("GREETING")
    if re.search(r"(とは(?:何|なに)?|って(?:何|なに)|の定義|どういう意味|何を意味)", q) and re.search(r"(とは|[?？]|意味|定義|って何)", q):
        return INTENTS.index("DEFINITION")
    if re.search(r"(好き(?:ですか|なの|？|\?)|嫌い(?:ですか|なの|？|\?)|好み|お気に入り)", q):
        return INTENTS.index("PREFERENCE")
    if re.search(r"(疲れました|疲れた|眠い|つらい|悲しい|嬉しい|寂しい|しんどい)", q) and not re.search(r"[?？]", q):
        return INTENTS.index("FEELING")
    if re.search(r"(教えてください|説明してください|おすすめ(?:は|を)|してください|してくれませんか)", q):
        return INTENTS.index("REQUEST")
    if re.search(r"(何ですか|なんですか|何でしょう|いつですか|どこですか|誰ですか|なぜ|どうして|化学組成|成分は|何が入って)", q):
        return INTENTS.index("FACT_QUESTION")
    return UNKNOWN_INTENT


class IntentRoPELanguageModel(RoPELanguageModel):
    """Same RoPE language model with extra d_model->intent_count classifier."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.intent_head = nn.Linear(self.d_model, len(INTENTS))

    def config(self):
        cfg = super().config()
        cfg["auxiliary_head"] = "dialogue_intent_v097"
        cfg["intent_labels"] = list(INTENTS)
        return cfg

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.hidden(ids))

    def forward_with_intent(self, ids: torch.Tensor, last_prompt_positions: torch.Tensor):
        """Auxiliary head sees only text up through the question/prompt.

        Causal attention prevents it from inspecting later reference answers.
        """
        h = self.hidden(ids)
        batch_size, length, _ = h.shape
        if (last_prompt_positions.ndim != 1 or len(last_prompt_positions) != batch_size
                or bool((last_prompt_positions < 0).any())
                or bool((last_prompt_positions >= length).any())):
            raise ValueError("Bad prompt position vector")
        indices = torch.arange(batch_size, device=ids.device)
        intent_logits = self.intent_head(h[indices, last_prompt_positions])
        return self.lm_head(h), intent_logits


def initialize_intent_model(base_checkpoint, *, device="cpu", seed=42):
    """Initialize head on a RoPE checkpoint, preserving *all* backbone weights."""
    from mha_rope_v096 import load_model_v096, checkpoint_architecture
    base, checkpoint = load_model_v096(
        base_checkpoint, torch.device(device), expected="rope"
    )
    cfg = checkpoint["config"]
    if cfg.get("auxiliary_head") is not None:
        raise ValueError("Source checkpoint already has an auxiliary head")
    # Avoid shifting the global seed used for the paired experiment.
    torch.manual_seed(seed)
    keys = ("vocab_size", "d_model", "num_layers",
            "hidden_dim", "causal", "context_length")
    model = IntentRoPELanguageModel(**{k: cfg[k] for k in keys if k in cfg})
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    if set(missing) != {"intent_head.weight", "intent_head.bias"} or unexpected:
        raise RuntimeError(f"Backbone mismatch: {missing} / {unexpected}")
    return model.to(device), checkpoint


def load_intent_model(checkpoint_path, device):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    cfg = checkpoint.get("config", {})
    if cfg.get("position_encoding") != "rope_v093" or cfg.get("auxiliary_head") != "dialogue_intent_v097":
        raise ValueError("Not a v0.9.7 single-head RoPE intent checkpoint")
    if tuple(cfg.get("intent_labels", [])) != INTENTS:
        raise ValueError("Intent class order mismatch")
    keys = ("vocab_size", "d_model", "num_layers",
            "hidden_dim", "causal", "context_length")
    model = IntentRoPELanguageModel(**{k: cfg[k] for k in keys if k in cfg})
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device), checkpoint
