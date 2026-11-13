"""Prepare controlled MHA RoPE v0.9.6 initial state from SAME fresh v0.9.5 Medium init.

Do not convert a trained checkpoint. Each weight tensor is copied bit-for-bit,
but the forward algorithm changes, so new training is mandatory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from mha_rope_v096 import prepare_mha_initial_checkpoint


def run(args):
    if args.output.resolve() in {args.source.resolve(), args.report.resolve()}:
        raise ValueError("Output must not overwrite inputs or report")
    if args.report.resolve() == args.source.resolve():
        raise ValueError("Report must not overwrite source")
    if not args.overwrite and (args.output.exists() or args.report.exists()):
        raise FileExistsError("Output/report exists; do not overwrite checkpoint without approval")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = prepare_mha_initial_checkpoint(args.source, args.output, heads=args.heads)
    report = {
        "version": "v0.9.6",
        "source_init": str(args.source),
        "mha_init": str(args.output),
        "num_heads": args.heads,
        "head_dim": model.d_model // args.heads,
        "parameters": model.parameter_count,
        "config": model.config(),
        "training_steps": 0,
        "note": "All weights copied from fresh v0.9.5 Medium RoPE initial state. "
                "Attention is NOT functionally equivalent; retrain before evaluation. "
                "Existing learned checkpoints were not modified.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Create a 4-head RoPE model from fresh 1-head RoPE init")
    p.add_argument("--source", type=Path, default=Path("model/model-rag-rope-medium-init-v0.9.5.pt"))
    p.add_argument("--output", type=Path, default=Path("model/model-rag-mha4-medium-init-v0.9.6.pt"))
    p.add_argument("--report", type=Path, default=Path("results/mha4_medium_init_v096.json"))
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
