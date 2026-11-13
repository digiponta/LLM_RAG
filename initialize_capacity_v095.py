"""v0.9.5 model-capacity study: reproducible, separately saved fresh RoPE initializations.

Because model shapes differ, an apples-to-apples capacity study should train
SMALL and MEDIUM/LARGE from fresh seeds on EXACTLY the same data. The already
fine-tuned v0.9.3 model is a separate warm-start reference, NOT the matched
scratch-training baseline. Does not overwrite saved checkpoints by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROFILES = {
    "small": {"d_model": 64, "num_layers": 2, "hidden_dim": 256},
    "medium": {"d_model": 128, "num_layers": 4, "hidden_dim": 512},
    "large": {"d_model": 128, "num_layers": 6, "hidden_dim": 512},
}


def build_config(profile: str, vocab_size: int, context_length: int) -> dict:
    if profile not in PROFILES:
        raise ValueError("Unknown capacity profile")
    if vocab_size <= 4 or context_length < 32:
        raise ValueError("Invalid vocabulary/context")
    return {
        "vocab_size": vocab_size,
        **PROFILES[profile],
        "context_length": context_length,
        "causal": True,
    }


def run(args):
    import torch
    from rope_model_v093 import RoPELanguageModel
    import sys
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from tokenizer import Tokenizer
    if args.checkpoint.resolve() == args.report.resolve():
        raise ValueError("Checkpoint/report must differ")
    if args.checkpoint.resolve() == args.tokenizer.resolve() or args.report.resolve() == args.tokenizer.resolve():
        raise ValueError("Cannot overwrite tokenizer")
    if not args.overwrite:
        for p in (args.checkpoint, args.report):
            if p.exists():
                raise FileExistsError(f"Already exists: {p}")
    tokenizer = Tokenizer.load(str(args.tokenizer))
    cfg = build_config(args.profile, tokenizer.vocab_size, args.context_length)
    torch.manual_seed(args.seed)
    model = RoPELanguageModel(**cfg)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Save a new RoPE checkpoint (no optimizer, no training yet).
    model.save_checkpoint(str(args.checkpoint), epoch=0, loss=None)
    report = {
        "version": "v0.9.5", "profile": args.profile,
        "init": "fresh_random_not_transplanted",
        "seed": args.seed, "parameter_count": model.parameter_count,
        "config": model.config(), "checkpoint": str(args.checkpoint),
        "training_steps": 0,
        "comparison_warning": (
            "For a controlled architecture-capacity ablation, train ALL profiles "
            "from fresh checkpoints using the same data, optimization schedule, "
            "seeds and Test split; do not compare a scratch-trained large model "
            "directly against pretrained v0.9.3 as a pure capacity effect."),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    p = argparse.ArgumentParser(description="Initialize a fresh RoPE model capacity profile")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--profile", choices=tuple(PROFILES), default="medium")
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path, default=Path("model/model-rag-rope-medium-init-v0.9.5.pt"))
    p.add_argument("--report", type=Path, default=Path("results/rope_medium_init_v095.json"))
    p.add_argument("--overwrite", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
