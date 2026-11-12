"""Prepare *synthetic quotation-style* dialogue examples from the Nagato corpus.

These are retrieval-to-quotation demonstrations, not human-authored question
answers. Outputs are never used to modify model weights automatically.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

TOPICS = ("宇宙", "時間", "量子", "粒子", "観測", "空間", "情報", "論理", "エネルギ", "人間")


def build_examples(lines: list[str]) -> list[dict]:
    examples = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("@NagatFakeBot Tweet Messages"):
            continue
        topic = next((t for t in TOPICS if t in line), None)
        if not topic:
            continue
        examples.append({
            "id": f"nagato-{idx + 1:04d}",
            "question": f"{topic}に関する元の記述を引用してください。",
            "answer": f"資料では、こう記されている。\n「{line}」",
            "topic": topic,
            "source": "data-nagato.txt",
            "source_line": idx + 1,
            "example_type": "synthetic_extractive_qa",
        })
    return examples


def split_examples(examples: list[dict], seed: int = 42, val_fraction: float = .15):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    values = list(examples)
    random.Random(seed).shuffle(values)
    n = max(1, round(len(values) * val_fraction)) if len(values) > 1 else 0
    return values[n:], values[:n]


def write_jsonl(path: Path, rows: list[dict], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Prepare extractive dialogue examples, without learning")
    p.add_argument("--input", type=Path, default=Path("data/knowledge/data-nagato.txt"))
    p.add_argument("--train-output", type=Path, default=Path("data/dialogue/nagato_synthetic_train_v06.jsonl"))
    p.add_argument("--val-output", type=Path, default=Path("data/dialogue/nagato_synthetic_val_v06.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=.15)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.train_output.resolve() == args.val_output.resolve():
        p.error("train-output and val-output must be different")
    # Validate both destinations before any write.
    for dest in (args.train_output, args.val_output):
        if dest.exists() and not args.overwrite:
            p.error(f"Output already exists: {dest}; use --overwrite to replace.")
    lines = args.input.read_text(encoding="utf-8-sig").splitlines()
    rows = build_examples(lines)
    if len(rows) < 2:
        p.error("Corpus does not have enough topic-matched posts.")
    train, val = split_examples(rows, args.seed, args.val_fraction)
    write_jsonl(args.train_output, train, args.overwrite)
    write_jsonl(args.val_output, val, args.overwrite)
    print(f"Prepared synthetic quotation examples: total={len(rows)} train={len(train)} validation={len(val)}")
    print(f"train={args.train_output} validation={args.val_output}")
    print("NOTE: No model training was performed.")


if __name__ == "__main__":
    main()
