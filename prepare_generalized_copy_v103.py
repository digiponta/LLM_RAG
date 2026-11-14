"""v0.10.3 diversified exact-copy SFT dataset.

Natural excerpts use original Nagato TRAIN vs VALIDATION source-line partitions.
Synthetic 10-character combinations are generated from the TRAIN-only character
inventory, with disjoint generated strings across train and development.
The final Nagato TEST partition is never opened by this generator.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from prepare_knowledge_binding_v100 import excerpt_from_v099
from prepare_unified_sft_v099 import write_jsonl
from train_real_dialogue_v082 import load_dialogue_rows, encode_dialogue

QUESTION = "次の資料を一文字も変えずに、そのまま書き写してください。"


def originals(rows):
    return [(r["conversation_id"], excerpt_from_v099(r))
            for r in rows if r.get("example_type") == "source_attributed_qa"]


def alphabet(examples):
    pool = sorted({c for _, text in examples for c in text if not c.isspace()})
    if len(pool) < 10:
        raise ValueError("Insufficient character variety")
    return pool


def make_row(text, *, split, kind, origin, index):
    return {
        "id": f"copy-v103:{split}:{kind}:{index}",
        "conversation_id": f"copy-v103:{split}:{origin}",
        "question": QUESTION,
        "context": "資料:" + text,
        "answer": text,
        "source_kind": "nagato_corpus" if kind == "natural" else "synthetic_composition",
        "copy_kind": kind,
        "copy_width": len(text),
        "example_type": "generalized_copy",
        "review_state": "SOURCE_QUOTE_UNVERIFIED" if kind == "natural" else "SYNTHETIC_TEST_STRING",
    }


def choose(examples, pool, tokenizer, *, split, count, width=10, seed=42,
           context_length=256, banned=None):
    if count < 1 or width < 1:
        raise ValueError("count and width must be positive")
    available = [(cid, text) for cid, text in examples if len(text) >= width]
    if not available:
        raise ValueError("No original excerpts long enough")
    rng = random.Random(seed)
    selected, used = [], set(banned or ())
    stats = Counter()
    # Interleave natural substrings and independently composed random strings.
    for attempt in range(max(5000, count * 100)):
        kind = "natural" if attempt % 2 == 0 else "synthetic"
        if kind == "natural":
            origin, source = rng.choice(available)
            start = rng.randrange(len(source) - width + 1)
            text = source[start:start + width]
        else:
            origin = f"synth-{attempt}"
            text = "".join(rng.choice(pool) for _ in range(width))
        if text in used:
            stats["duplicate"] += 1
            continue
        row = make_row(text, split=split, kind=kind, origin=origin, index=attempt)
        pair, status = encode_dialogue(row, tokenizer, context_length, .10,
                                       min_answer_tokens=1)
        if pair is None:
            stats[status] += 1
            continue
        selected.append(row)
        used.add(text)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only prepared {len(selected)}/{count} tokenizer-compatible cases")
    stats["accepted"] = len(selected)
    return selected, dict(stats)


def guard(train, dev):
    if {r["id"] for r in train} & {r["id"] for r in dev}:
        raise ValueError("Row ID overlap")
    if {r["answer"] for r in train} & {r["answer"] for r in dev}:
        raise ValueError("Exact answer leakage")
    train_origin = {r["conversation_id"] for r in train if r["copy_kind"] == "natural"}
    dev_origin = {r["conversation_id"] for r in dev if r["copy_kind"] == "natural"}
    if train_origin & dev_origin:
        raise ValueError("Original-line leakage")


def prepare(args):
    from tokenizer import Tokenizer
    tok = Tokenizer.load(str(args.tokenizer))
    output = {
        "train": args.output_dir / "copy_general_train_v103.jsonl",
        "development": args.output_dir / "copy_general_development_v103.jsonl",
        "report": args.output_dir / "copy_general_report_v103.json",
    }
    paths = [args.nagato_train, args.nagato_validation, args.tokenizer, *output.values()]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Input/output path collision")
    if any(p.exists() for p in output.values()) and not args.overwrite:
        raise FileExistsError("Outputs already exist")
    train_src = originals(load_dialogue_rows(args.nagato_train, 0, args.seed))
    dev_src = originals(load_dialogue_rows(args.nagato_validation, 0, args.seed))
    if set(x[0] for x in train_src) & set(x[0] for x in dev_src):
        raise ValueError("Source-line partition overlap")
    pool = alphabet(train_src)
    train, stats_train = choose(train_src, pool, tok, split="train", count=args.train_count,
                                width=args.width, seed=args.seed,
                                context_length=args.context_length)
    dev, stats_dev = choose(dev_src, pool, tok, split="development",
                            count=args.development_count, width=args.width,
                            seed=args.seed + 1, context_length=args.context_length,
                            banned={r["answer"] for r in train})
    guard(train, dev)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output["train"], train)
    write_jsonl(output["development"], dev)
    report = {
        "version": "v0.10.3", "width_chars": args.width,
        "train_count": len(train), "development_count": len(dev),
        "train_by_kind": dict(Counter(r["copy_kind"] for r in train)),
        "development_by_kind": dict(Counter(r["copy_kind"] for r in dev)),
        "train_stats": stats_train, "development_stats": stats_dev,
        "training_alphabet_chars": len(pool),
        "final_nagato_test_used": False,
        "limitation": "Copy targets are already in input; success is copying, not factuality or retrieval.",
    }
    output["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Diversified 10-character copy SFT data")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--nagato-train", type=Path, default=Path(
        "data/dialogue/unified_v099/nagato_train_v099.jsonl"))
    p.add_argument("--nagato-validation", type=Path, default=Path(
        "data/dialogue/unified_v099/nagato_validation_v099.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("data/dialogue/copy_general_v103"))
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--train-count", type=int, default=2000)
    p.add_argument("--development-count", type=int, default=200)
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
