"""LLM_RAG v0.10.6 — Knowledge/Persona/Dialogue retention, same-model comparisons.

Compares frozen v0.10.0, v0.10.3 and v0.10.5 on source-disjoint
DEVELOPMENT/VALIDATION rows, NOT final TEST. Nagato statements are
source-attributed, unverified, potentially fictional; this is not fact checking.

Per-task answer-only teacher-forced NLL is calculated separately; sampled
greedy generation is a structural proxy, not a human-rated quality score.
No source text or answers in reports without explicit --include-text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from task_conditioning_v108 import tagged

TASKS = ("nagato_knowledge", "nagato_persona", "real_persona", "evidence_binding")
CHECKPOINTS = ("binding_v100", "copy_v103", "position_v105")


def choose_rows(nagato, unified, binding):
    """Keep task populations disjoint in type (not necessarily source topic)."""
    selected = {
        "nagato_knowledge": [r for r in nagato
                             if r.get("example_type") == "source_attributed_qa"
                             and r.get("source_kind") == "nagato_corpus"],
        "nagato_persona": [r for r in nagato
                           if r.get("example_type") == "persona_style_from_source"
                           and r.get("source_kind") == "nagato_corpus"],
        "real_persona": [r for r in unified
                         if r.get("source_kind", "real_persona_chat") == "real_persona_chat"],
        "evidence_binding": [r for r in binding
                             if r.get("example_type") == "evidence_direct"
                             and r.get("source_kind") == "nagato_corpus"],
    }
    if any(not selected[t] for t in TASKS):
        raise ValueError("One or more validation task groups have zero records")
    for task, rows in selected.items():
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{task}: duplicate IDs")
    return selected


def guard_validation_paths(paths):
    """Refuse known final test material by filename and overlapping paths."""
    resolved = [Path(x).resolve() for x in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Dataset paths must be distinct")
    for item in paths:
        name = Path(item).name.lower()
        if "test" in name or "holdout" in name or "final" in name:
            raise ValueError(f"Do not use final/test/holdout material: {item}")


def sample_rows(rows, n, seed):
    if n < 1:
        raise ValueError("Sample must be >= 1")
    return random.Random(seed).sample(rows, min(len(rows), n))


def summarize(samples, names):
    output = {}
    for name in names:
        exact = sum(row[name]["exact_match"] for row in samples)
        output[name] = {
            "generated_samples": len(samples),
            "exact_response_matches": exact,
            "exact_response_rate": exact / len(samples) if samples else None,
            "eos_termination_fraction": (
                sum(row[name]["ended_by_eos"] for row in samples) / len(samples)
                if samples else None),
            "mean_output_characters": (
                sum(row[name]["generated_chars"] for row in samples) / len(samples)
                if samples else None),
            "mean_answer_length_abs_error": (
                sum(abs(row[name]["generated_chars"] - row["reference_chars"])
                    for row in samples) / len(samples) if samples else None),
        }
    return output


def score(args):
    import torch
    from torch.utils.data import DataLoader
    from mha_rope_v096 import load_model_v096
    from tokenizer import Tokenizer
    from prepare_real_persona_v081 import compact_prompt
    from train_real_dialogue_v082 import (
        load_dialogue_rows, encode_dialogue, collate_dialogue, evaluate_answer_nll)

    datasets = (args.nagato_validation, args.unified_validation, args.binding_validation)
    candidate_path = getattr(args, "candidate_checkpoint", None)
    paths = [*datasets, args.tokenizer, args.binding_checkpoint,
             args.copy_checkpoint, args.position_checkpoint, args.output]
    if candidate_path is not None:
        paths.append(candidate_path)
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Input/output path collision")
    guard_validation_paths(datasets)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("Report exists; use a fresh output path")
    if args.max_rows < 1 or args.generations < 0 or args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("Invalid evaluation limits")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    tok = Tokenizer.load(str(args.tokenizer))
    models = {}
    checkpoints = [
        ("binding_v100", args.binding_checkpoint),
        ("copy_v103", args.copy_checkpoint),
        ("position_v105", args.position_checkpoint),
    ]
    if candidate_path is not None:
        checkpoints.append(("multitask_v107", candidate_path))
    for name, path in checkpoints:
        model, _ = load_model_v096(path, device, expected="rope")
        if model.vocab_size != tok.vocab_size:
            raise ValueError(f"{name}: tokenizer mismatch")
        models[name] = model.eval()
    context_limit = min(model.context_length for model in models.values())
    tasks = choose_rows(
        load_dialogue_rows(args.nagato_validation, 0, args.seed),
        load_dialogue_rows(args.unified_validation, 0, args.seed),
        load_dialogue_rows(args.binding_validation, 0, args.seed),
    )
    results = {}
    for offset, task in enumerate(TASKS):
        subset = sample_rows(tasks[task], args.max_rows, args.seed + offset)
        accepted, pairs, rejects = [], [], Counter()
        for row in subset:
            pair, reason = encode_dialogue(row, tok, context_limit,
                                            args.max_unk_fraction)
            if pair is None:
                rejects[reason] += 1
            else:
                pairs.append(pair)
                accepted.append(row)
        if not pairs:
            raise ValueError(f"No usable {task} validation records")
        loader = DataLoader(
            pairs, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda batch: collate_dialogue(batch, tok.pad_id))
        scores = {
            name: {"answer_nll": evaluate_answer_nll(model, loader, device)}
            for name, model in models.items() if name != "multitask_v107"
                or not getattr(args, "candidate_task_tag", False)
        }
        if candidate_path is not None and getattr(args, "candidate_task_tag", False):
            tagged_pairs = []
            for row in accepted:
                tagged_pair, reason = encode_dialogue(
                    tagged(row, task), tok, context_limit, args.max_unk_fraction)
                if tagged_pair is None:
                    raise ValueError(f"Tagged candidate context invalid: {task}/{reason}")
                tagged_pairs.append(tagged_pair)
            tagged_loader = DataLoader(
                tagged_pairs, batch_size=args.batch_size, shuffle=False,
                collate_fn=lambda batch: collate_dialogue(batch, tok.pad_id))
            scores["multitask_v107"] = {
                "answer_nll": evaluate_answer_nll(
                    models["multitask_v107"], tagged_loader, device)}

        generations = []
        for row in accepted[:args.generations]:
            prompt = tok.encode(compact_prompt(row), add_bos=True)
            entry = {
                "example_sha256": hashlib.sha256(row["id"].encode("utf-8")).hexdigest()[:16],
                "reference_chars": len(row["answer"]),
                "origin_sha256": hashlib.sha256(row["conversation_id"].encode("utf-8")).hexdigest()[:16],
            }
            if args.include_text:
                entry.update(question=row["question"], context=row.get("context", ""),
                             reference_answer=row["answer"])
            for name, model in models.items():
                candidate_prompt = (tok.encode(compact_prompt(tagged(row, task)), add_bos=True)
                                    if name == "multitask_v107" and getattr(args, "candidate_task_tag", False)
                                    else prompt)
                with torch.inference_mode():
                    raw = model.generate(
                        candidate_prompt, max_new_tokens=args.max_new_tokens,
                        eos_id=tok.eos_id, temperature=0, top_k=None,
                        repetition_penalty=1.0)
                tail = raw[len(candidate_prompt):]
                generated = tok.decode(tail, skip_special_tokens=True)
                entry[name] = {
                    "exact_match": generated == row["answer"],
                    "generated_chars": len(generated),
                    "ended_by_eos": bool(tail and tail[-1] == tok.eos_id),
                    "output_sha256": hashlib.sha256(generated.encode("utf-8")).hexdigest()[:16],
                }
                if args.include_text:
                    entry[name]["generated"] = generated
            generations.append(entry)
        summary = summarize(generations, models.keys())
        for name in models:
            scores[name].update(summary[name])
        results[task] = {
            "source_records": len(tasks[task]),
            "selected": len(subset),
            "usable": len(pairs),
            "rejected": dict(rejects),
            "models": scores,
            "samples": generations,
        }
        print(json.dumps({"task": task, "selected": len(subset),
                          "usable": len(pairs), "models": scores}, ensure_ascii=False))
    report = {
        "version": "v0.10.6",
        "validation_only": True,
        "checkpoints": {name: str(path) for name, path in checkpoints},
        "tasks": results,
        "text_included": bool(args.include_text),
        "candidate_task_tag": bool(getattr(args, "candidate_task_tag", False)),
        "warnings": (
            "Each task's answer NLL is comparable ACROSS MODELS for the same task, "
            "not across differently formatted tasks. Exact generative matches are "
            "a strict lexical proxy, not conversational quality, personality "
            "alignment, or scientific truth. Nagato facts are unverified source "
            "claims. Binding evaluation gives the reference as oracle evidence. "
            "Nagato and Persona validation were part of prior model selection; "
            "results are DEVELOPMENT, not independent final holdout. "
            "Do not compare absolute NLL across checkpoints trained under "
            "different tasks as though it measures a single ability."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"report": str(args.output),
                      "tasks": {k: {n: v["models"][n]["answer_nll"]
                                    for n in models} for k, v in results.items()}},
                     ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Nagato knowledge/persona/dialogue retention")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--nagato-validation", type=Path, default=Path(
        "data/dialogue/unified_v099/nagato_validation_v099.jsonl"))
    p.add_argument("--unified-validation", type=Path, default=Path(
        "data/dialogue/unified_v099/unified_validation_v099.jsonl"))
    p.add_argument("--binding-validation", type=Path, default=Path(
        "data/dialogue/binding_v100/binding_validation_v100.jsonl"))
    p.add_argument("--binding-checkpoint", type=Path, default=Path(
        "model/model-rag-binding-v0.10.0.pt"))
    p.add_argument("--copy-checkpoint", type=Path, default=Path(
        "model/model-rag-copy-general-v0.10.3.pt"))
    p.add_argument("--position-checkpoint", type=Path, default=Path(
        "model/model-rag-position-v0.10.5.pt"))
    p.add_argument("--output", type=Path, default=Path(
        "results/nagato_retention_v106.json"))
    p.add_argument("--candidate-checkpoint", type=Path, default=None,
                   help="Optional candidate checkpoint")
    p.add_argument("--candidate-task-tag", action="store_true",
                   help="Apply task markers only to candidate; baseline prompt unchanged")
    p.add_argument("--max-rows", type=int, default=200)
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-unk-fraction", type=float, default=.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.gpu_dir / "model/tokenizer.json"
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    score(args)


if __name__ == "__main__":
    main()
