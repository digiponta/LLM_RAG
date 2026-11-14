"""v0.10.7 deterministic balanced multi-task SFT from already separated sources.

No TEST files or raw data-nagato.txt opened. This is a controlled experiment,
not a validated single-model knowledge/persona release.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from evaluate_nagato_retention_v106 import guard_validation_paths
from prepare_unified_sft_v099 import write_jsonl
from train_real_dialogue_v082 import encode_dialogue, load_dialogue_rows

TASKS = ("real_persona", "nagato_knowledge", "nagato_persona",
         "evidence_binding", "position", "copy")
WEIGHTS = {"real_persona": 40, "nagato_knowledge": 20,
           "nagato_persona": 15, "evidence_binding": 15,
           "position": 5, "copy": 5}
FILES = {
    "real_persona": ("data/dialogue/unified_v099/unified_train_v099.jsonl",
                     "data/dialogue/unified_v099/unified_validation_v099.jsonl"),
    "nagato_knowledge": ("data/dialogue/unified_v099/nagato_train_v099.jsonl",
                         "data/dialogue/unified_v099/nagato_validation_v099.jsonl"),
    "nagato_persona": ("data/dialogue/unified_v099/nagato_train_v099.jsonl",
                       "data/dialogue/unified_v099/nagato_validation_v099.jsonl"),
    "evidence_binding": ("data/dialogue/binding_v100/binding_train_v100.jsonl",
                         "data/dialogue/binding_v100/binding_validation_v100.jsonl"),
    "position": ("data/dialogue/position_v105/position_train_v105.jsonl",
                 "data/dialogue/position_v105/position_development_v105.jsonl"),
    "copy": ("data/dialogue/copy_general_v103/copy_general_train_v103.jsonl",
             "data/dialogue/copy_general_v103/copy_general_development_v103.jsonl"),
}


def source_type(task, row):
    if task == "real_persona":
        return row.get("source_kind", "real_persona_chat") == "real_persona_chat"
    if task == "nagato_knowledge":
        return row.get("source_kind") == "nagato_corpus" and row.get("example_type") == "source_attributed_qa"
    if task == "nagato_persona":
        return row.get("source_kind") == "nagato_corpus" and row.get("example_type") == "persona_style_from_source"
    if task == "evidence_binding":
        return row.get("source_kind") == "nagato_corpus" and row.get("example_type") == "evidence_direct"
    if task == "position":
        return row.get("example_type") == "position_extract"
    if task == "copy":
        return row.get("example_type") == "generalized_copy"
    raise ValueError(f"Unrecognized task: {task}")


def canonical_origin(row):
    """Recover full original line origin for natural copy/position rows.

    Other samples use their own conversation ID; source splits are independently
    enforced by the input pipelines as well.
    """
    source = row["conversation_id"]
    original = source
    for prefix in ("position-v105:train:", "position-v105:development:",
                   "copy-v103:train:", "copy-v103:development:"):
        if source.startswith(prefix):
            source = source[len(prefix):]
    # Natural quotes carry the underlying nagato-v099 line identity.
    # Synthetic IDs can recur across splits (e.g. synth-1); preserve split tag.
    return source if source.startswith("nagato-v099:") else original


def filter_rows(rows, task, tok, context_length):
    good, skipped = [], Counter()
    for row in rows:
        if not source_type(task, row):
            continue
        if task == "position" and tok.unk_id in tok.encode(row["answer"]):
            skipped["unknown_target"] += 1
            continue
        pair, status = encode_dialogue(
            row, tok, context_length, .10,
            min_answer_tokens=1 if task == "position" else 2)
        skipped[status] += 1
        if pair is not None:
            good.append(row)
    if not good:
        raise ValueError(f"{task}: no valid rows")
    return good, dict(skipped)


def quota(total, weights=WEIGHTS):
    if total < len(TASKS) or total % 20 != 0:
        raise ValueError("Total must be a multiple of 20 and at least 20")
    if set(weights) != set(TASKS) or sum(weights.values()) != 100 or any(
            type(w) is not int or w < 1 for w in weights.values()):
        raise ValueError("Weights must be positive integers totaling 100")
    # Largest remainder, deterministic tie order.
    raw = {task: total * weights[task] / 100 for task in TASKS}
    numbers = {task: int(raw[task]) for task in TASKS}
    left = total - sum(numbers.values())
    for task in sorted(TASKS, key=lambda k: (-(raw[k]-numbers[k]), TASKS.index(k)))[:left]:
        numbers[task] += 1
    return numbers


def sample_balanced(groups, total, *, seed=42, max_repeat=8, split="train", weights=WEIGHTS):
    amounts = quota(total, weights)
    if max_repeat < 1:
        raise ValueError("max_repeat must be >=1")
    result, audit = [], {}
    for offset, task in enumerate(TASKS):
        candidates = sorted(groups[task], key=lambda r: r["id"])
        if len({r["id"] for r in candidates}) != len(candidates):
            raise ValueError(f"{task}: duplicate row ID")
        if not candidates:
            raise ValueError(f"{task}: empty group")
        required = amounts[task]
        if required > max_repeat * len(candidates):
            raise ValueError(
                f"{task}: quota {required} needs >{max_repeat} repeats for "
                f"{len(candidates)} unique examples; reduce --total or raise --max-repeat deliberately")
        rng = random.Random(seed + offset)
        candidates = candidates.copy()
        rng.shuffle(candidates)
        selected = []
        for n in range(required):
            original = candidates[n % len(candidates)]
            selected.append({
                **original, "id": f"balanced-v107:{split}:{task}:{n}",
                "conversation_id": f"balanced-v107:{split}:{task}:" + original["conversation_id"],
                "multitask": task, "original_example_id": original["id"],
                "original_conversation_id": original["conversation_id"],
                "original_source_kind": original.get("source_kind", "real_persona_chat"),
            })
        result.extend(selected)
        audit[task] = {"available_unique_rows": len(candidates),
                       "selected": required, "unique_selected": min(required, len(candidates)),
                       "max_repetition": (required + len(candidates)-1) // len(candidates)}
    random.Random(seed + 777).shuffle(result)
    return result, audit


def separation(train, validation):
    for task in TASKS:
        tr = [r for r in train if r["multitask"] == task]
        va = [r for r in validation if r["multitask"] == task]
        if ({r["original_example_id"] for r in tr}
                & {r["original_example_id"] for r in va}):
            raise ValueError(f"{task}: example overlap")
    tr_origins = {canonical_origin({"conversation_id": r["original_conversation_id"]}) for r in train}
    va_origins = {canonical_origin({"conversation_id": r["original_conversation_id"]}) for r in validation}
    overlap = tr_origins & va_origins
    if overlap:
        raise ValueError(f"Cross-partition original conversation overlap: {len(overlap)}")
    tr_texts = {r["context"] for r in train if r["multitask"] in ("position", "copy")}
    va_texts = {r["context"] for r in validation if r["multitask"] in ("position", "copy")}
    if tr_texts & va_texts:
        raise ValueError("Duplicate copy/position source strings across splits")


def prepare(args):
    from tokenizer import Tokenizer
    if args.validation_total < 20 or args.train_total < 20:
        raise ValueError("Expected positive sample sizes")
    train_files = {task: Path(FILES[task][0]) for task in TASKS}
    dev_files = {task: Path(FILES[task][1]) for task in TASKS}
    guard_validation_paths(list(set(dev_files.values())))
    targets = {
        "train": args.output_dir / "multitask_train_v107.jsonl",
        "validation": args.output_dir / "multitask_validation_v107.jsonl",
        "report": args.output_dir / "multitask_report_v107.json",
    }
    inputs = {p.resolve() for p in [*train_files.values(), *dev_files.values(), args.tokenizer]}
    if inputs & {p.resolve() for p in targets.values()}:
        raise ValueError("Destination overwrites source")
    if not args.overwrite and any(x.exists() for x in targets.values()):
        raise FileExistsError("Existing output; refuse overwrite")
    tok = Tokenizer.load(str(args.tokenizer))
    loaded = {}
    groups = {"train": {}, "validation": {}}
    skipped = {}
    for task in TASKS:
        for split, path in (("train", train_files[task]), ("validation", dev_files[task])):
            key = path.resolve()
            if key not in loaded:
                loaded[key] = load_dialogue_rows(path, 0, args.seed)
            accepted, stats = filter_rows(loaded[key], task, tok, args.context_length)
            groups[split][task] = accepted
            skipped[f"{split}:{task}"] = stats
    train, audit_train = sample_balanced(groups["train"], args.train_total,
                                          seed=args.seed, max_repeat=args.max_repeat, split="train")
    dev, audit_dev = sample_balanced(groups["validation"], args.validation_total,
                                      seed=args.seed+1, max_repeat=args.max_repeat, split="validation")
    separation(train, dev)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(targets["train"], train)
    write_jsonl(targets["validation"], dev)
    report = {
        "version": "v0.10.7", "weights_percent": WEIGHTS,
        "train_count": len(train), "validation_count": len(dev),
        "train_task_audit": audit_train, "validation_task_audit": audit_dev,
        "filter_stats": skipped, "final_test_opened": False,
        "limitation": "Dataset rows are oversampled deterministically, not independent observations. "
        "Nagato source claims are not verified facts; balanced ratios are initial hypotheses.",
    }
    targets["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Balanced source-separated 6-task SFT mixer")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("data/dialogue/multitask_v107"))
    p.add_argument("--train-total", type=int, default=4000)
    p.add_argument("--validation-total", type=int, default=400)
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
