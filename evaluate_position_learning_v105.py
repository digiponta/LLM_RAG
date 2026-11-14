"""LLM_RAG v0.10.5: paired position-extraction generalization evaluator.

Compares frozen v0.10.3 and newly trained v0.10.5 on the same DEVELOPMENT
original and counterfactual evidence. Includes exact one-character answer,
gold first-token score, gold-conditioned EOS probability and paired correct
transition. Generated answers are not printed unless --include-text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from evaluate_evidence_position_v104 import position_row, swap_pairs
from evaluate_generation_diagnostics_v098 import first_token_diagnostic
from prepare_real_persona_v081 import compact_prompt
from train_real_dialogue_v082 import load_dialogue_rows
from task_conditioning_v108 import tagged


def decode_one(model, tokenizer, row, limit=4):
    import torch
    prefix = tokenizer.encode(compact_prompt(row), add_bos=True)
    gold = tokenizer.encode(row["answer"])
    if not gold:
        raise ValueError("Missing encoded gold token")
    if len(prefix) + len(gold) >= model.context_length:
        raise ValueError("Gold response exceeds context")
    first = first_token_diagnostic(model, prefix, gold)
    # Teacher-forced EOS probability is distinct from free generation EOS.
    with torch.inference_mode():
        device = next(model.parameters()).device
        seq = torch.tensor([prefix + gold], dtype=torch.long, device=device)
        logits = model(seq)[0, -1].float()
        eos_probability = float(logits.softmax(dim=-1)[tokenizer.eos_id].item())
        tokens = model.generate(prefix, max_new_tokens=limit,
                                eos_id=tokenizer.eos_id, temperature=0,
                                top_k=None, repetition_penalty=1.0)
    tail = tokens[len(prefix):]
    response = tokenizer.decode(tail, skip_special_tokens=True)
    return {
        "exact": response == row["answer"],
        "first_token_correct": first["correct"],
        "first_token_probability": first["target_probability"],
        "gold_eos_probability": eos_probability,
        "ended_by_eos": bool(tail and tail[-1] == tokenizer.eos_id),
        "generated_chars": len(response),
        "gold_token_count": len(gold),
        "generated_hash": hashlib.sha256(response.encode("utf-8")).hexdigest()[:16],
        "text": response,
    }


def aggregate(entries, model):
    by = defaultdict(list)
    for r in entries:
        keys = [
            "all", "position_" + str(r["position"]),
            r["kind"], r["kind"] + "_" + r["variant"],
            r["variant"],
        ]
        for key in keys:
            by[key].append(r[model])
    result = {}
    for key, scores in by.items():
        n = len(scores)
        result[key] = {
            "cases": n,
            "exact_rate": sum(s["exact"] for s in scores) / n,
            "first_token_accuracy": sum(s["first_token_correct"] for s in scores) / n,
            "mean_gold_eos_probability": sum(s["gold_eos_probability"] for s in scores) / n,
            "eos_termination_fraction": sum(s["ended_by_eos"] for s in scores) / n,
            "mean_generated_chars": sum(s["generated_chars"] for s in scores) / n,
        }
    return result


def pair_metrics(pairs, model):
    """Score answer change, gold change, and *correct tracking* separately."""
    if not pairs:
        raise ValueError("No paired comparisons")
    changed_output = changed_gold = correct_both = changed_correct = 0
    for old, new in pairs:
        changed = old["gold"] != new["gold"]
        changed_gold += changed
        changed_output += old[model]["generated_hash"] != new[model]["generated_hash"]
        correct_both += old[model]["exact"] and new[model]["exact"]
        changed_correct += changed and old[model]["exact"] and new[model]["exact"]
    n = len(pairs)
    return {
        "pairs": n,
        "gold_change_fraction": changed_gold / n,
        "output_change_fraction": changed_output / n,
        "both_original_and_swapped_correct_fraction": correct_both / n,
        "changed_gold_and_both_correct_fraction": changed_correct / n,
        "changed_gold_and_both_correct_given_changed_gold": (
            changed_correct / changed_gold if changed_gold else None),
    }


def evaluate(args):
    import torch
    from tokenizer import Tokenizer
    from mha_rope_v096 import load_model_v096

    paths = (args.baseline, args.tuned, args.dataset, args.tokenizer)
    if args.output.resolve() in {x.resolve() for x in paths}:
        raise ValueError("Output overlaps an input")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("Output already exists")
    if args.sample < 0 or args.max_new_tokens < 1:
        raise ValueError("Invalid evaluation options")
    if not args.positions or set(args.positions) - {1, 5, 10} or len(args.positions) != len(set(args.positions)):
        raise ValueError("Positions must be unique from 1,5,10")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    tok = Tokenizer.load(str(args.tokenizer))
    models = {}
    for name, path in (("baseline", args.baseline), ("tuned", args.tuned)):
        mdl, _ = load_model_v096(path, device, expected="rope")
        if mdl.vocab_size != tok.vocab_size:
            raise ValueError("Tokenizer mismatch")
        models[name] = mdl.eval()
    sources = load_dialogue_rows(args.dataset, args.sample, args.seed)
    originals_and_donors = swap_pairs(sources)
    rows, comparisons = [], []
    for original, donor in originals_and_donors:
        for pos in args.positions:
            o = position_row(original, pos)
            s = position_row(original, pos, evidence=donor["answer"])
            pair = []
            for variant, source in (("original", o), ("swapped", s)):
                result = {"case_id": hashlib.sha256(
                              (original["id"] + ":" + str(pos)).encode("utf-8")
                          ).hexdigest()[:20],
                          "kind": original["copy_kind"],
                          "position": pos, "variant": variant, "gold": source["answer"]}
                for name, model in models.items():
                    input_row = tagged(source, "position") if name == "tuned" and getattr(args, "tuned_task_tag", False) else source
                    result[name] = decode_one(model, tok, input_row, args.max_new_tokens)
                rows.append(result)
                pair.append(result)
            comparisons.append(pair)
    groups = {name: aggregate(rows, name) for name in models}
    changes = {name: pair_metrics(comparisons, name) for name in models}
    if not args.include_text:
        for row in rows:
            row.pop("gold")
            for name in models:
                row[name].pop("text")
    report = {
        "version": "v0.10.5", "dataset": str(args.dataset), "cases": len(rows),
        "paired_cases": len(comparisons),
        "model_checkpoints": {"baseline": str(args.baseline), "tuned": str(args.tuned)},
        "groups": groups, "counterfactuals": changes, "details": rows,
        "text_included": bool(args.include_text),
        "tuned_task_tag": bool(getattr(args, "tuned_task_tag", False)),
        "limitation": (
            "The requested character exists verbatim in evidence. "
            "Development results are not independent final test results. "
            "EOS probability is teacher-forced after the gold answer; "
            "free-generation exact matching and EOS termination are separate. "
            "Swapped gold may equal original at a given position; paired success "
            "on genuinely changed-gold cases is reported separately."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"cases": len(rows), "groups": groups, "counterfactuals": changes,
                      "report": str(args.output)}, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Compare trained position extraction and EOS")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--baseline", type=Path, default=Path(
        "model/model-rag-copy-general-v0.10.3.pt"))
    p.add_argument("--tuned", type=Path, default=Path(
        "model/model-rag-position-v0.10.5.pt"))
    p.add_argument("--dataset", type=Path, default=Path(
        "data/dialogue/copy_general_v103/copy_general_development_v103.jsonl"))
    p.add_argument("--output", type=Path, default=Path(
        "results/position_generalization_v105.json"))
    p.add_argument("--sample", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--positions", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--tuned-task-tag", action="store_true",
                   help="Use [TASK:POSITION] only on tuned model, matching tagged SFT training")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.gpu_dir / "model/tokenizer.json"
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    evaluate(args)

if __name__ == "__main__":
    main()
