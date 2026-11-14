"""v0.10.3: per-kind exact-copy evaluation for frozen baseline and tuned models.

Same paired examples for both; no training or private text in default report.
A 10-character copy benchmark does not establish evidence retrieval or reasoning.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from evaluate_evidence_copy_v101 import score_copy
from evaluate_generation_diagnostics_v098 import first_token_diagnostic
from prepare_real_persona_v081 import compact_prompt
from train_real_dialogue_v082 import load_dialogue_rows


def validate_copy_rows(rows):
    seen = set()
    for row in rows:
        if row.get("example_type") != "generalized_copy":
            raise ValueError("Wrong evaluation task")
        if row.get("copy_kind") not in ("natural", "synthetic"):
            raise ValueError("Unknown copy_kind")
        if row["answer"] != row.get("context", "")[3:] or not row["context"].startswith("資料:"):
            raise ValueError("Answer must be exactly provided context evidence")
        if row["answer"] in seen:
            raise ValueError("Repeated target")
        seen.add(row["answer"])


def summarize(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["kind"]].append(row[key])
        grouped["all"].append(row[key])
    return {
        k: {
            "cases": len(xs),
            "exact_rate": sum(x["exact_match"] for x in xs) / len(xs),
            "first_token_accuracy": sum(x["first_token_correct"] for x in xs) / len(xs),
            "mean_prefix_coverage": sum(x["prefix_coverage"] for x in xs) / len(xs),
            "mean_position_accuracy": sum(x["position_accuracy"] for x in xs) / len(xs),
            "mean_generated_chars": sum(x["generated_chars"] for x in xs) / len(xs),
        } for k, xs in grouped.items()
    }


def evaluate(args):
    import torch
    from tokenizer import Tokenizer
    from mha_rope_v096 import load_model_v096

    sources = [args.baseline, args.tuned, args.dataset, args.tokenizer]
    if args.output.resolve() in {p.resolve() for p in sources}:
        raise ValueError("Output would overwrite input")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("Output report already exists")
    if args.sample < 0 or args.max_new_tokens < 1:
        raise ValueError("Invalid sample or generation length")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    tok = Tokenizer.load(str(args.tokenizer))
    models = {}
    for name, path in (("baseline", args.baseline), ("tuned", args.tuned)):
        model, _ = load_model_v096(path, device, expected="rope")
        if model.vocab_size != tok.vocab_size:
            raise ValueError("Tokenizer mismatch")
        models[name] = model.eval()
    # The default 0 means ALL examples in this development evaluation.
    rows = load_dialogue_rows(args.dataset, args.sample, args.seed)
    validate_copy_rows(rows)
    records = []
    for row in rows:
        prompt = tok.encode(compact_prompt(row), add_bos=True)
        answer = tok.encode(row["answer"])
        if max(len(prompt) + len(answer), len(prompt) + 1) > min(m.context_length for m in models.values()):
            raise ValueError("Unencodable or over-context copy case")
        entry = {"case_id": row["id"], "kind": row["copy_kind"],
                 "reference_chars": len(row["answer"])}
        if args.include_text:
            entry["reference"] = row["answer"]
        for name, model in models.items():
            first = first_token_diagnostic(model, prompt, answer)
            with torch.inference_mode():
                output = model.generate(prompt, max_new_tokens=max(args.max_new_tokens,
                        len(answer) + 8), eos_id=tok.eos_id, temperature=0,
                        top_k=None, repetition_penalty=1.0)
            generated = output[len(prompt):]
            response = tok.decode(generated, skip_special_tokens=True)
            metric = score_copy(row["answer"], response,
                                eos=bool(generated and generated[-1] == tok.eos_id))
            metric["first_token_correct"] = first["correct"]
            if args.include_text:
                metric["generated"] = response
            entry[name] = metric
        records.append(entry)
    report = {
        "version": "v0.10.3", "dataset": str(args.dataset),
        "count": len(records),
        "models": {name: summarize(records, name) for name in models},
        "cases": records,
        "text_included": args.include_text,
        "qualification": (
            "All targets are present verbatim in the context. "
            "This evaluates literal source reproduction, not knowledge or retrieval. "
            "Natural and synthetic composition subsets are reported separately."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"count": len(records), "models": report["models"],
                      "report": str(args.output)}, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Paired exact-copy generation evaluation")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--baseline", type=Path, default=Path(
        "model/model-rag-binding-v0.10.0.pt"))
    p.add_argument("--tuned", type=Path, default=Path(
        "model/model-rag-copy-general-v0.10.3.pt"))
    p.add_argument("--dataset", type=Path, default=Path(
        "data/dialogue/copy_general_v103/copy_general_development_v103.jsonl"))
    p.add_argument("--output", type=Path, default=Path(
        "results/copy_general_development_v103.json"))
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=32)
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
