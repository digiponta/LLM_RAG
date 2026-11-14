"""LLM_RAG v0.10.2: prepare source-disjoint 10-character COPY learning probe.

Use ONLY the v0.9.9 Nagato train and validation splits. Keep Nagato TEST
untouched for later final assessment. Never mix line fragments across partitions.
Generated JSONL stays local (covered by data/dialogue/ .gitignore).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from evaluate_evidence_copy_v101 import copy_row
from prepare_unified_sft_v099 import write_jsonl
from train_real_dialogue_v082 import load_dialogue_rows, encode_dialogue


def select_examples(rows, tok, *, count=16, width=10, context_length=256,
                    max_unk_fraction=.10, seed=42):
    """Unique original source lines, with actual SFT acceptance checks."""
    if count < 1 or width < 1:
        raise ValueError("count and width must be positive")
    candidates = list(rows)
    random.Random(seed).shuffle(candidates)
    output, origins = [], set()
    for source in candidates:
        if source.get("example_type") != "source_attributed_qa":
            continue
        origin = source["conversation_id"]
        if origin in origins:
            continue
        item = copy_row(source, width)
        if item is None:
            continue
        pair, reason = encode_dialogue(item, tok, context_length,
                                       max_unk_fraction, min_answer_tokens=1)
        if pair is None:
            continue
        output.append(item)
        origins.add(origin)
        if len(output) >= count:
            break
    if not output:
        raise ValueError("No usable source-copy examples")
    return output


def guard_sources(train, validation):
    t = {r["conversation_id"] for r in train}
    v = {r["conversation_id"] for r in validation}
    if t & v:
        raise ValueError("Source-line leakage between train and development")
    if len(t) != len(train) or len(v) != len(validation):
        raise ValueError("Duplicate source line within a partition")


def prepare(args):
    from tokenizer import Tokenizer
    tok = Tokenizer.load(str(args.tokenizer))
    inputs = (args.nagato_train, args.nagato_validation, args.tokenizer)
    outputs = {
        "train": args.output_dir / "copy_train_v102.jsonl",
        "development": args.output_dir / "copy_development_v102.jsonl",
        "report": args.output_dir / "copy_prepare_report_v102.json",
    }
    if len({p.resolve() for p in (*inputs, *outputs.values())}) != len(inputs) + len(outputs):
        raise ValueError("Input or output files overlap")
    if not args.overwrite and any(p.exists() for p in outputs.values()):
        raise FileExistsError("Output exists; use a new output directory")
    train_sources = load_dialogue_rows(args.nagato_train, 0, args.seed)
    dev_sources = load_dialogue_rows(args.nagato_validation, 0, args.seed)
    train = select_examples(train_sources, tok, count=args.train_count, width=args.width,
                            context_length=args.context_length, seed=args.seed)
    development = select_examples(dev_sources, tok, count=args.development_count,
                                  width=args.width, context_length=args.context_length,
                                  seed=args.seed + 1)
    guard_sources(train, development)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(outputs["train"], train)
    write_jsonl(outputs["development"], development)
    report = {
        "version": "v0.10.2",
        "copy_width_chars": args.width,
        "train_count": len(train),
        "development_count": len(development),
        "train_sources": str(args.nagato_train),
        "development_sources": str(args.nagato_validation),
        "final_nagato_test_used": False,
        "limits": "Small controlled experiment; do not infer general copy ability from train memorization.",
    }
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Prepare 10-character copy-learning probe")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--nagato-train", type=Path, default=Path(
        "data/dialogue/unified_v099/nagato_train_v099.jsonl"))
    p.add_argument("--nagato-validation", type=Path, default=Path(
        "data/dialogue/unified_v099/nagato_validation_v099.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("data/dialogue/copy_probe_v102"))
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--train-count", type=int, default=16)
    p.add_argument("--development-count", type=int, default=8)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    prepare(args)


if __name__ == "__main__":
    main()
