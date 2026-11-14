"""v0.10.5 position-question SFT data from v0.10.3 TRAIN and DEVELOPMENT.

Each source row spawns three examples whose answer is one character at
position 1, 5, or 10. Original source partitions remain intact, and Nagato
final TEST is never read. Data are local, .gitignored.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from evaluate_evidence_position_v104 import position_row
from evaluate_generalized_copy_v103 import validate_copy_rows
from prepare_unified_sft_v099 import write_jsonl
from train_real_dialogue_v082 import load_dialogue_rows, encode_dialogue

POSITIONS = (1, 5, 10)


def transform(rows, tok, *, split, positions=POSITIONS, context_length=256):
    validate_copy_rows(rows)
    if not positions or len(set(positions)) != len(positions) or any(
            type(n) is not int or n < 1 for n in positions):
        raise ValueError("Invalid positions")
    results, counts = [], Counter()
    for source in rows:
        for position in positions:
            if position > len(source["answer"]):
                raise ValueError("Source shorter than requested position")
            item = position_row(source, position)
            item["id"] = f"position-v105:{split}:{source['id']}:{position}"
            item["conversation_id"] = f"position-v105:{split}:{source['conversation_id']}"
            item["example_type"] = "position_extract"
            item["copy_kind"] = source["copy_kind"]
            item["char_position"] = position
            item["source_row_id"] = source["id"]
            encoded, reason = encode_dialogue(item, tok, context_length, .10,
                                               min_answer_tokens=1)
            if encoded is None:
                counts[reason] += 1
                continue
            counts[f"position_{position}_accepted"] += 1
            results.append(item)
    if not results:
        raise ValueError("No accepted position examples")
    return results, dict(counts)


def guard(train, dev):
    if {r["id"] for r in train} & {r["id"] for r in dev}:
        raise ValueError("Overlapping row IDs")
    if {r["conversation_id"] for r in train} & {r["conversation_id"] for r in dev}:
        raise ValueError("Overlapping source lines")
    if {r["context"] for r in train} & {r["context"] for r in dev}:
        raise ValueError("Overlapping source text")


def prepare(args):
    from tokenizer import Tokenizer
    paths = {
        "train": args.output_dir / "position_train_v105.jsonl",
        "development": args.output_dir / "position_development_v105.jsonl",
        "report": args.output_dir / "position_prepare_report_v105.json",
    }
    inputs = [args.copy_train, args.copy_development, args.tokenizer]
    if len({p.resolve() for p in [*inputs, *paths.values()]}) != len(inputs) + len(paths):
        raise ValueError("File path collision")
    if not args.overwrite and any(p.exists() for p in paths.values()):
        raise FileExistsError("Output already exists")
    tok = Tokenizer.load(str(args.tokenizer))
    train_rows = load_dialogue_rows(args.copy_train, 0, args.seed)
    dev_rows = load_dialogue_rows(args.copy_development, 0, args.seed)
    if {r["conversation_id"] for r in train_rows} & {r["conversation_id"] for r in dev_rows}:
        raise ValueError("Overlapping source lines in input")
    if {r["answer"] for r in train_rows} & {r["answer"] for r in dev_rows}:
        raise ValueError("Overlapping source text in input")
    train, stats_train = transform(train_rows, tok, split="train",
                                   positions=args.positions, context_length=args.context_length)
    dev, stats_dev = transform(dev_rows, tok, split="development",
                               positions=args.positions, context_length=args.context_length)
    guard(train, dev)
    if len(train) != len(train_rows) * len(args.positions) or len(dev) != len(dev_rows) * len(args.positions):
        raise ValueError("Some position records were rejected: reduce unknown fraction or inspect input")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["train"], train)
    write_jsonl(paths["development"], dev)
    report = {
        "version": "v0.10.5", "positions": args.positions,
        "train_examples": len(train), "development_examples": len(dev),
        "train_by_kind": dict(Counter(r["copy_kind"] for r in train)),
        "development_by_kind": dict(Counter(r["copy_kind"] for r in dev)),
        "train_stats": stats_train, "development_stats": stats_dev,
        "nagato_final_test_used": False,
        "qualification": "The gold character is already supplied in context. This tests position extraction only.",
    }
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Prepare source-disjoint position-character SFT")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--copy-train", type=Path, default=Path(
        "data/dialogue/copy_general_v103/copy_general_train_v103.jsonl"))
    p.add_argument("--copy-development", type=Path, default=Path(
        "data/dialogue/copy_general_v103/copy_general_development_v103.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("data/dialogue/position_v105"))
    p.add_argument("--positions", type=int, nargs="+", default=[1, 5, 10])
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
