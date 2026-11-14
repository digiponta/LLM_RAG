"""v0.10.9 task-tagged position-ratio sweep with FIXED development distribution.

Use v0.9.9/v0.10.3/v0.10.5 TRAIN source pools; never read final TEST.
Unlike v0.10.7, development source rows and mixture are exactly the SAME
pre-existing v0.10.8 task_validation_v108.jsonl across all ratio variants.
This isolates increased position exposure while retaining an identical
development selection criterion (not an independent final test).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from prepare_balanced_multitask_v107 import (
    TASKS, FILES, filter_rows, sample_balanced, separation)
from prepare_task_conditioned_v108 import transform
from prepare_unified_sft_v099 import write_jsonl
from train_real_dialogue_v082 import load_dialogue_rows

WEIGHT_SWEEP = {
    5: {"real_persona": 40, "nagato_knowledge": 20,
        "nagato_persona": 15, "evidence_binding": 15,
        "position": 5, "copy": 5},
    10: {"real_persona": 38, "nagato_knowledge": 19,
         "nagato_persona": 14, "evidence_binding": 14,
         "position": 10, "copy": 5},
    20: {"real_persona": 33, "nagato_knowledge": 17,
         "nagato_persona": 13, "evidence_binding": 12,
         "position": 20, "copy": 5},
    30: {"real_persona": 29, "nagato_knowledge": 15,
         "nagato_persona": 11, "evidence_binding": 10,
         "position": 30, "copy": 5},
}


def balanced_positions(rows, count, seed=42):
    """Ensure each requested character position is represented in training."""
    positions = {1: [], 5: [], 10: []}
    for row in rows:
        pos = row.get("char_position")
        if pos not in positions:
            raise ValueError("Unexpected position record")
        positions[pos].append(row)
    total = sum(len(v) for v in positions.values())
    if total == 0 or count < 3:
        raise ValueError("Not enough position samples")
    chosen = []
    for index, p in enumerate((1, 5, 10)):
        candidates = sorted(positions[p], key=lambda x: x["id"])
        random.Random(seed + p).shuffle(candidates)
        quota = count // 3 + (1 if index < count % 3 else 0)
        if quota > len(candidates):
            raise ValueError("Insufficient unique examples per position")
        chosen.extend(candidates[:quota])
    return chosen


def prepare(args):
    from tokenizer import Tokenizer
    if args.position_percent not in WEIGHT_SWEEP:
        raise ValueError("Use --position-percent 5,10,20,30")
    if args.train_total < 20 or args.train_total % 20:
        raise ValueError("Train total must be a positive multiple of 20")
    if args.max_repeat < 1:
        raise ValueError("max-repeat must be >=1")
    if args.validation.name.lower().find("test") >= 0 or "holdout" in args.validation.name.lower():
        raise ValueError("Refuse TEST as development")
    weights = WEIGHT_SWEEP[args.position_percent]
    output = {
        "train": args.output_dir / f"task_train_pos{args.position_percent}_v109.jsonl",
        "validation": args.output_dir / "task_validation_fixed_v109.jsonl",
        "report": args.output_dir / f"position_ratio_report_pos{args.position_percent}_v109.json",
    }
    inputs = [*{Path(FILES[t][0]) for t in TASKS}, args.validation, args.tokenizer]
    if {p.resolve() for p in inputs} & {p.resolve() for p in output.values()}:
        raise ValueError("Output overwrites an input")
    if not args.overwrite and any(x.exists() for x in output.values()):
        raise FileExistsError("Output exists; use a new output directory")
    tok = Tokenizer.load(str(args.tokenizer))
    fixed_dev = load_dialogue_rows(args.validation, 0, args.seed)
    if any(r.get("multitask") not in TASKS or not r.get("task_marker") for r in fixed_dev):
        raise ValueError("Fixed development must be v0.10.8 tagged and multitask-labeled")
    sources, groups = {}, {}
    for task in TASKS:
        path = Path(FILES[task][0]).resolve()
        if path not in sources:
            sources[path] = load_dialogue_rows(path, 0, args.seed)
        groups[task], _ = filter_rows(sources[path], task, tok, args.context_length)

    # The default v0.10.7 sampler is deterministic, capped and source-separated.
    # Unlike unstratified 5% sampling, ensure positions 1/5/10 each receive
    # about one third of the position quota.
    position_quota = args.train_total * weights["position"] // 100
    groups["position"] = balanced_positions(groups["position"], position_quota, args.seed)

    mixed, audit = sample_balanced(
        groups, args.train_total, seed=args.seed, split="train",
        max_repeat=args.max_repeat, weights=weights)
    # The selected position examples are already balanced and unique.
    # All mixed samples remain subject to exact encoder verification.
    separation(mixed, fixed_dev)
    tagged_train, stats = transform(mixed, tok, args.context_length)
    separation(tagged_train, fixed_dev)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output["train"], tagged_train)
    write_jsonl(output["validation"], fixed_dev)
    actual_position = Counter(r["char_position"] for r in tagged_train
                              if r["multitask"] == "position")
    report = {
        "version": "v0.10.9", "position_percent": args.position_percent,
        "weights": weights, "train_total": len(tagged_train),
        "fixed_validation_count": len(fixed_dev),
        "train_by_task": dict(Counter(r["multitask"] for r in tagged_train)),
        "position_by_char_index": dict(actual_position),
        "repetition_audit": audit, "encode_stats": stats,
        "fixed_validation_source": str(args.validation),
        "nagato_final_test_used": False,
        "note": "This is a position-ratio sweep. All variants must use identical "
        "v0.10.8 validation rows, same checkpoint and train hyperparameters. "
        "The per-task counts and chosen TRAIN rows differ by design.",
    }
    output["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Prepare tagged controlled position ratio sweep")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--validation", type=Path, default=Path(
        "data/dialogue/task_conditioned_v108/task_validation_v108.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path(
        "data/dialogue/task_ratio_pos10_v109"))
    p.add_argument("--position-percent", type=int, default=10)
    p.add_argument("--train-total", type=int, default=4000)
    p.add_argument("--max-repeat", type=int, default=8)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.gpu_dir / "model/tokenizer.json"
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    prepare(args)


if __name__ == "__main__":
    main()
