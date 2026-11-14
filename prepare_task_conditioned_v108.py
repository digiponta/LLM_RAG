"""Prepare tagged versions of v0.10.7 balanced TRAIN/DEVELOPMENT.

Same source rows and quotas as untagged control. Never touch source files or
original final Test, no tokenizer/architecture change.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from task_conditioning_v108 import tagged
from prepare_unified_sft_v099 import write_jsonl
from train_real_dialogue_v082 import load_dialogue_rows, encode_dialogue
from prepare_balanced_multitask_v107 import separation, TASKS


def transform(rows, tok, context_length):
    tagged_rows = []
    stats = Counter()
    for row in rows:
        task = row.get("multitask")
        if task not in TASKS:
            raise ValueError("Input must be v0.10.7 task-labeled sample")
        item = tagged(row)
        pair, status = encode_dialogue(item, tok, context_length, .10,
                                       min_answer_tokens=1 if task == "position" else 2)
        stats[f"{task}:{status}"] += 1
        if pair is None:
            raise ValueError(
                f"Task-tagged row became unencodable; {task}/{status}; "
                "do not silently change task sampling weights")
        tagged_rows.append(item)
    return tagged_rows, dict(stats)


def prepare(args):
    from tokenizer import Tokenizer
    inputs = (args.train, args.validation, args.tokenizer)
    outputs = {
        "train": args.output_dir / "task_train_v108.jsonl",
        "validation": args.output_dir / "task_validation_v108.jsonl",
        "report": args.output_dir / "task_prepare_report_v108.json",
    }
    if args.train.resolve() == args.validation.resolve():
        raise ValueError("Training and validation must be different")
    if {p.resolve() for p in inputs} & {p.resolve() for p in outputs.values()}:
        raise ValueError("Output conflicts with input")
    if not args.overwrite and any(p.exists() for p in outputs.values()):
        raise FileExistsError("Task outputs exist already")
    tok = Tokenizer.load(str(args.tokenizer))
    tr = load_dialogue_rows(args.train, 0, 42)
    va = load_dialogue_rows(args.validation, 0, 42)
    if any(r.get("multitask") not in TASKS for r in tr + va):
        raise ValueError("Missing v0.10.7 multitask labels")
    separation(tr, va)
    train, train_stats = transform(tr, tok, args.context_length)
    validation, val_stats = transform(va, tok, args.context_length)
    separation(train, validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(outputs["train"], train)
    write_jsonl(outputs["validation"], validation)
    report = {
        "version": "v0.10.8", "train_total": len(train),
        "validation_total": len(validation),
        "train_by_task": dict(Counter(r["multitask"] for r in train)),
        "validation_by_task": dict(Counter(r["multitask"] for r in validation)),
        "train_encodings": train_stats, "validation_encodings": val_stats,
        "data_only": "Original data and checkpoints preserved; source TEST not used",
    }
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Add task markers to balanced mixed SFT")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--train", type=Path, default=Path(
        "data/dialogue/multitask_v107/multitask_train_v107.jsonl"))
    p.add_argument("--validation", type=Path, default=Path(
        "data/dialogue/multitask_v107/multitask_validation_v107.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path(
        "data/dialogue/task_conditioned_v108"))
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.gpu_dir / "model/tokenizer.json"
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    prepare(args)


if __name__ == "__main__":
    main()
