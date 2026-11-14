"""LLM_RAG v0.10.4: diagnostic position probes + counterfactual evidence swap.

Frozen v0.10.0/v0.10.3 models; source-disjoint v0.10.3 DEVELOPMENT rows
only. Ask for character 1, 5, or 10; compare original and swapped evidence,
holding the question and position fixed. Free generation and teacher-forced
first-token rank are reported separately.

Important: Neither model was trained with the position-question instruction.
Poor performance CANNOT establish an attention/position-encoding defect.
Character offsets and tokenizer token offsets are not interchangeable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from evaluate_generation_diagnostics_v098 import first_token_diagnostic
from prepare_real_persona_v081 import compact_prompt
from train_real_dialogue_v082 import load_dialogue_rows
from evaluate_generalized_copy_v103 import validate_copy_rows


def position_question(position):
    if not isinstance(position, int) or isinstance(position, bool) or position < 1:
        raise ValueError("Position must be positive integer")
    return f"資料の{position}文字目だけを、他の文字を付けずに答えてください。"


def position_row(source, position, evidence=None):
    if source.get("example_type") != "generalized_copy":
        raise ValueError("Position input requires generalized copy data")
    text = source["answer"] if evidence is None else evidence
    if not isinstance(text, str) or position < 1 or position > len(text):
        raise ValueError("Requested character does not exist")
    return {
        "id": f"position:{source['id']}:{position}",
        "conversation_id": source["conversation_id"],
        "question": position_question(position),
        "context": "資料:" + text,
        "answer": text[position - 1],
    }


def swap_pairs(rows):
    """Deterministic one-to-one in-kind cyclic derangement of source evidence.

    No source answers ever modified. Returns list[(original, donor)].
    If there is only one row in a category, reject rather than self-swap.
    """
    validate_copy_rows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["copy_kind"]].append(row)
    pairs = []
    for kind, items in groups.items():
        if len(items) < 2:
            raise ValueError(f"Need two different records to swap kind={kind}")
        for index, row in enumerate(items):
            donor = items[(index + 1) % len(items)]
            if row["answer"] == donor["answer"]:
                raise ValueError("Swapped evidence must differ")
            pairs.append((row, donor))
    return pairs


def summarize(entries, model):
    buckets = defaultdict(list)
    for case in entries:
        buckets[f"position_{case['position']}"].append(case[model])
        buckets[f"{case['kind']}_original" if case["variant"] == "original"
                else f"{case['kind']}_swapped"].append(case[model])
        buckets[case["variant"]].append(case[model])
    result = {}
    for key, values in buckets.items():
        n = len(values)
        result[key] = {
            "cases": n,
            "exact_character_fraction": sum(v["exact_character"] for v in values) / n,
            "first_token_accuracy": sum(v["first_token_correct"] for v in values) / n,
            "mean_gold_first_token_rank": sum(v["first_token_rank"] for v in values) / n,
            "mean_gold_first_token_probability": sum(v["first_token_probability"] for v in values) / n,
            "single_token_target_fraction": sum(v["gold_token_count"] == 1 for v in values) / n,
        }
    return result


def evaluate(args):
    import torch
    from tokenizer import Tokenizer
    from mha_rope_v096 import load_model_v096

    if not args.positions or any(n not in (1, 5, 10) for n in args.positions):
        raise ValueError("Choose positions from 1, 5, 10")
    if len(set(args.positions)) != len(args.positions) or args.max_new_tokens < 1:
        raise ValueError("Invalid position or generation settings")
    if args.sample < 0:
        raise ValueError("sample must be >= 0")
    sources = [args.dataset, args.baseline, args.tuned, args.tokenizer]
    if args.output.resolve() in {p.resolve() for p in sources}:
        raise ValueError("Output would overwrite an input")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("Report already exists")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    models = {}
    for name, path in (("baseline", args.baseline), ("tuned", args.tuned)):
        model, _ = load_model_v096(path, device, expected="rope")
        if model.vocab_size != tokenizer.vocab_size:
            raise ValueError("Tokenizer/model mismatch")
        models[name] = model.eval()
    rows = load_dialogue_rows(args.dataset, args.sample, args.seed)
    pairs = swap_pairs(rows)
    records = []
    changed_output = {name: 0 for name in models}
    paired_total = 0

    for original, donor in pairs:
        for position in args.positions:
            if min(len(original["answer"]), len(donor["answer"])) < position:
                continue
            original_case = position_row(original, position)
            swapped_case = position_row(original, position, evidence=donor["answer"])
            entry_by_variant = []
            for variant, row in (("original", original_case), ("swapped", swapped_case)):
                prompt = tokenizer.encode(compact_prompt(row), add_bos=True)
                gold = tokenizer.encode(row["answer"])
                if not gold:
                    raise ValueError("Tokenization produced no answer tokens")
                if len(prompt) + len(gold) > min(m.context_length for m in models.values()):
                    raise ValueError("Over-context example")
                entry = {
                    "case_id": hashlib.sha256(
                        (original["id"] + ":" + str(position)).encode("utf-8")
                    ).hexdigest()[:20],
                    "kind": original["copy_kind"],
                    "position": position,
                    "variant": variant,
                    "gold_token_count": len(gold),
                    "changed_gold_character": original_case["answer"] != swapped_case["answer"],
                }
                if args.include_text:
                    entry.update({"evidence": row["context"], "gold": row["answer"]})
                for name, model in models.items():
                    diagnostic = first_token_diagnostic(model, prompt, gold)
                    with torch.inference_mode():
                        tokens = model.generate(
                            prompt, max_new_tokens=args.max_new_tokens,
                            eos_id=tokenizer.eos_id, temperature=0,
                            top_k=None, repetition_penalty=1.0)
                    output = tokenizer.decode(tokens[len(prompt):],
                                              skip_special_tokens=True)
                    entry[name] = {
                        "exact_character": output == row["answer"],
                        "first_token_correct": diagnostic["correct"],
                        "first_token_rank": diagnostic["target_rank"],
                        "first_token_probability": diagnostic["target_probability"],
                        "gold_token_count": len(gold),
                        "output_hash": hashlib.sha256(output.encode("utf-8")).hexdigest()[:16],
                    }
                    if args.include_text:
                        entry[name]["generated"] = output
                records.append(entry)
                entry_by_variant.append(entry)
            if len(entry_by_variant) == 2:
                paired_total += 1
                for name in models:
                    changed_output[name] += (
                        entry_by_variant[0][name]["output_hash"]
                        != entry_by_variant[1][name]["output_hash"])
    if not records:
        raise ValueError("No compatible position probes")
    result = {
        "version": "v0.10.4",
        "dataset": str(args.dataset),
        "positions_char_1_based": args.positions,
        "records": len(records),
        "counterfactual_pairs": paired_total,
        "counterfactual_output_change_rate": {
            name: n / paired_total for name, n in changed_output.items()
        },
        "models": {name: summarize(records, name) for name in models},
        "cases": records,
        "text_included": bool(args.include_text),
        "limitations": (
            "Zero-shot position questions were NOT in the training curriculum. "
            "Failure does not imply deficient attention or positional encoding. "
            "Evidence swapping changes the gold target; an output change alone "
            "does not establish correctness. Position is a Unicode character index, "
            "not a tokenizer-token index; multi-token gold targets are reported. "
            "Test split remains untouched; no private source text is written by default."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({
        "records": len(records),
        "counterfactual_pairs": paired_total,
        "counterfactual_output_change_rate": result["counterfactual_output_change_rate"],
        "models": result["models"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description="Frozen evidence-position and swap diagnostics")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--dataset", type=Path, default=Path(
        "data/dialogue/copy_general_v103/copy_general_development_v103.jsonl"))
    p.add_argument("--baseline", type=Path, default=Path(
        "model/model-rag-binding-v0.10.0.pt"))
    p.add_argument("--tuned", type=Path, default=Path(
        "model/model-rag-copy-general-v0.10.3.pt"))
    p.add_argument("--output", type=Path, default=Path(
        "results/evidence_position_probe_v104.json"))
    p.add_argument("--positions", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--sample", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.gpu_dir / "model/tokenizer.json"
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    evaluate(args)


if __name__ == "__main__":
    main()
