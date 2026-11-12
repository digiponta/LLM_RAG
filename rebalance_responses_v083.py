"""v0.8.3: Analyze generic-answer concentration and prepare balanced TRAIN subset.

Does not touch validation/test or the original train file. No external I/O.
This is exact-answer frequency balancing, not a semantic quality classifier.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", "", value).strip("。！？!?、,. 　")


def response_bias(rows: list[dict], top_k: int = 10, expose_text: bool = False) -> dict:
    if not rows:
        raise ValueError("Empty rows")
    counts = Counter(normalize_answer(row["answer"]) for row in rows)
    counts.pop("", None)
    if not counts:
        raise ValueError("No nonempty answer strings")
    total = sum(counts.values())
    popular = counts.most_common(top_k)
    return {
        "sample_count": total,
        "distinct_normalized_answers": len(counts),
        "most_frequent_answer_share": popular[0][1] / total,
        "top_10_answer_share": sum(n for _, n in counts.most_common(10)) / total,
        "singletons": sum(n == 1 for n in counts.values()),
        "top_answers": [
            {**({"answer": answer} if expose_text else
                {"answer_hash": hashlib.sha256(answer.encode("utf-8")).hexdigest()[:12]}),
             "count": count, "share": count / total}
            for answer, count in popular
        ],
        "scope": "Exact normalized-answer frequency, not semantic equivalence or correctness",
    }


def balanced_sample(rows: list[dict], limit: int, max_per_answer: int,
                    seed: int) -> tuple[list[dict], dict]:
    if limit < 1 or max_per_answer < 1:
        raise ValueError("limit and max_per_answer must be positive")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    counts = Counter()
    selected = []
    for row in shuffled:
        key = normalize_answer(row["answer"])
        if not key or counts[key] >= max_per_answer:
            continue
        selected.append(row)
        counts[key] += 1
        if len(selected) == limit:
            break
    return selected, {"selected": len(selected), "requested": limit,
                      "max_per_answer": max_per_answer, "source_rows": len(rows),
                      "selection_shortfall": max(0, limit - len(selected))}


def main():
    parser = argparse.ArgumentParser(description="Assess answer repetition and write balanced train-only JSONL")
    parser.add_argument("--train", type=Path, default=Path("data/dialogue/real_persona_v081/real_persona_train_v081.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/dialogue/real_persona_v083/train_balanced_10000.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("results/response_bias_v083.json"))
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--max-per-answer", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expose-text", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.sample < 1:
        parser.error("--sample must be >=1")
    if len({args.train.resolve(), args.output.resolve(), args.report.resolve()}) != 3:
        parser.error("Input and output paths must be different")
    for path in (args.output, args.report):
        if path.exists() and not args.overwrite:
            parser.error(f"Existing output: {path} (choose another path or --overwrite)")
    # Import only here so no data is loaded when utility functions are tested.
    from train_real_dialogue_v082 import load_dialogue_rows
    rows = load_dialogue_rows(args.train, 0, args.seed)
    bias = response_bias(rows, expose_text=args.expose_text)
    chosen, selection = balanced_sample(rows, args.sample, args.max_per_answer, args.seed)
    if not chosen:
        raise ValueError("No usable selected rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in chosen:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"version": "v0.8.3", "response_bias": bias, "selection": selection,
              "selection_bias": response_bias(chosen, expose_text=args.expose_text),
              "note": ("Sampling changes training distribution only; validation and test are untouched. "
                       "This is an experimental frequency cap, not an independent test or a quality filter. "
                       "Raw chat examples are included in TRAIN output, keep local."),
              "output": str(args.output)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"response_bias": bias, "selection": selection,
                      "selection_bias": report["selection_bias"],
                      "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
