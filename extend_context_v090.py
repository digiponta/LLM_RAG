"""v0.9.0: repackage v0.8.4 64-token weights for a 256-token context.

The LLM_GPU v0.3 LanguageModel has NO absolute position embeddings:
context_length is a config/inference-window limit, not a parameter dimension.
Therefore all weights can be transferred with strict state-dict validation.
This does not establish competence at long contexts until fine-tuning/evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def validate_context(source: int, target: int) -> None:
    if not 16 <= source < target <= 2048:
        raise ValueError("Require 16 <= source_context < target_context <= 2048")


def transplant(args) -> dict:
    import torch

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")
    paths = [args.checkpoint, args.tokenizer]
    if args.output.resolve() in {p.resolve() for p in paths}:
        raise ValueError("Refusing to overwrite source checkpoint/tokenizer")
    if args.report.resolve() in {p.resolve() for p in [*paths, args.output]}:
        raise ValueError("Report must use a distinct path")
    if args.report.exists() and not args.overwrite:
        raise FileExistsError(f"Report already exists: {args.report}")

    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from model import LanguageModel
    from tokenizer import Tokenizer
    device = torch.device("cpu")  # avoid unnecessarily allocating GPU memory
    source, _ = LanguageModel.load_checkpoint(str(args.checkpoint), device=device)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    if source.vocab_size != tokenizer.vocab_size:
        raise ValueError("Tokenizer/checkpoint vocabulary mismatch")
    validate_context(source.context_length, args.context_length)

    source_config = source.config()
    if source_config.get("causal") is not True:
        raise ValueError("Expected causal decoder-only model")
    # Must be the known v0.3 architecture. This also guards against future
    # versions adding learnable positional parameters without an explicit copy.
    source_names = set(source.state_dict())
    if any("position" in name.lower() or "pos_embed" in name.lower()
           for name in source_names):
        raise ValueError("Positional weights present: manual conversion required")

    new_config = dict(source_config, context_length=args.context_length)
    target = LanguageModel(**new_config)
    target.load_state_dict(source.state_dict(), strict=True)
    if source.parameter_count != target.parameter_count:
        raise AssertionError("Parameter count changed unexpectedly")
    for name, tensor in source.state_dict().items():
        if not torch.equal(tensor, target.state_dict()[name]):
            raise AssertionError(f"Weight differs after transfer: {name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "llm-rag-context-extension-v0.9.0",
        "config": target.config(),
        "model_state_dict": target.state_dict(),
        "source_checkpoint": str(args.checkpoint),
        "extension_only": True,
        "training_steps": 0,
    }, args.output)
    report = {
        "version": "v0.9.0", "source": str(args.checkpoint),
        "output": str(args.output), "context_old": source.context_length,
        "context_new": target.context_length,
        "parameter_count_old": source.parameter_count,
        "parameter_count_new": target.parameter_count,
        "weights_identical": True, "training_steps": 0,
        "warning": ("Weights are transferred without positional embeddings. "
                    "Long-context ability is unverified and requires SFT/evaluation.")
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Expand LLM_GPU checkpoint context with identical weights")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--checkpoint", type=Path, default=Path("model/model-rag-real-dialogue-full-v0.8.4.pt"))
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--output", type=Path, default=Path("model/model-rag-context256-init-v0.9.0.pt"))
    p.add_argument("--report", type=Path, default=Path("results/context256_init_v090.json"))
    p.add_argument("--overwrite", action="store_true")
    transplant(p.parse_args())


if __name__ == "__main__":
    main()
