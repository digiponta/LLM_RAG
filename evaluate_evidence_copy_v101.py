"""v0.10.1 — frozen-model, length-controlled Evidence Copy diagnostics.

This is a diagnostic, NOT training or a new claim of semantic understanding.
Only v0.9.9 TEST source rows are used. Copy target is a literal 10/30/50
character prefix from an original Nagato attributed quote. We provide that
target inside the prompt and ask the LLM to reproduce it verbatim.
No local private text is written to reports unless --include-text is set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from prepare_knowledge_binding_v100 import excerpt_from_v099
from prepare_real_persona_v081 import compact_prompt
from train_real_dialogue_v082 import encode_dialogue, load_dialogue_rows
from evaluate_generation_diagnostics_v098 import first_token_diagnostic


def copy_row(source: dict, width: int) -> dict | None:
    """A copy task using exact source text, no synthesized factual answer."""
    if width < 1:
        raise ValueError("Copy width must be positive")
    if source.get("example_type") != "source_attributed_qa":
        return None
    quote = excerpt_from_v099(source)
    if len(quote) < width:
        return None
    excerpt = quote[:width]
    return {
        "id": f"copy-v101:{source['id']}:{width}",
        "conversation_id": source["conversation_id"],
        "question": "次の資料を一文字も変えずに、そのまま書き写してください。",
        "context": f"資料:{excerpt}",
        "answer": excerpt,
        "original_line_hash": source.get("original_line_hash", ""),
        "source_kind": "nagato_corpus",
        "review_state": "SOURCE_QUOTE_UNVERIFIED",
        "length_chars": width,
        "example_type": "evidence_copy",
    }


def assemble_cases(rows: list[dict], tokenizer, *,
                   widths: list[int], cases_per_width: int,
                   context_length: int, max_unk_fraction: float,
                   seed: int = 42) -> tuple[list[dict], dict]:
    """Choose repeatable, disjoint original source LINES per length.

    No source line contributes to more than one selected case across widths,
    preventing a handful of long lines from dominating all levels. No answer
    truncation: the existing real-dialogue encoder validates each full pair.
    """
    if not widths or len(set(widths)) != len(widths) or min(widths) < 1:
        raise ValueError("Widths must be distinct positive integers")
    if cases_per_width < 1:
        raise ValueError("cases_per_width must be positive")
    source_rows = [r for r in rows if r.get("example_type") == "source_attributed_qa"]
    rng = random.Random(seed)
    rng.shuffle(source_rows)
    selected, used_origins, stats = [], set(), Counter()
    # Descending lengths first avoids allocating a long-only source to 10 chars.
    for width in sorted(widths, reverse=True):
        count = 0
        for source in source_rows:
            origin = source["conversation_id"]
            if origin in used_origins:
                continue
            row = copy_row(source, width)
            if row is None:
                stats[f"{width}_insufficient_chars"] += 1
                continue
            encoded, reason = encode_dialogue(row, tokenizer, context_length,
                                              max_unk_fraction, min_answer_tokens=1)
            if encoded is None:
                stats[f"{width}_{reason}"] += 1
                continue
            selected.append(row)
            used_origins.add(origin)
            count += 1
            if count >= cases_per_width:
                break
        stats[f"{width}_selected"] = count
    if not selected:
        raise ValueError("No usable copy cases")
    return sorted(selected, key=lambda r: (r["length_chars"], r["id"])), dict(stats)


def longest_common_prefix(actual: str, expected: str) -> int:
    length = 0
    for a, b in zip(actual, expected):
        if a != b:
            break
        length += 1
    return length


def score_copy(expected: str, generated: str, *, eos: bool) -> dict:
    if not expected:
        raise ValueError("Reference must be nonempty")
    prefix = longest_common_prefix(generated, expected)
    positions_correct = sum(a == b for a, b in zip(generated, expected))
    return {
        "exact_match": generated == expected,
        "prefix_chars": prefix,
        "prefix_coverage": prefix / len(expected),
        "position_accuracy": positions_correct / len(expected),
        "sequence_similarity": SequenceMatcher(None, expected, generated,
                                               autojunk=False).ratio(),
        "generated_chars": len(generated),
        "length_error": len(generated) - len(expected),
        "stopped_eos": bool(eos),
    }


def summarize(cases: list[dict], model_name: str, widths: list[int]) -> dict:
    summary = {}
    for width in widths:
        selected = [r[model_name] for r in cases if r["length_chars"] == width]
        if not selected:
            summary[str(width)] = {"cases": 0}
            continue
        n = len(selected)
        summary[str(width)] = {
            "cases": n,
            "exact_rate": sum(r["exact_match"] for r in selected) / n,
            "first_token_accuracy": sum(r["first_token_correct"] for r in selected) / n,
            "mean_first_token_rank": sum(r["first_token_rank"] for r in selected) / n,
            "mean_prefix_coverage": sum(r["prefix_coverage"] for r in selected) / n,
            "mean_position_accuracy": sum(r["position_accuracy"] for r in selected) / n,
            "mean_sequence_similarity": sum(r["sequence_similarity"] for r in selected) / n,
            "mean_generated_chars": sum(r["generated_chars"] for r in selected) / n,
            "eos_fraction": sum(r["stopped_eos"] for r in selected) / n,
        }
    return summary


def evaluate(args):
    import torch
    from tokenizer import Tokenizer
    from mha_rope_v096 import load_model_v096

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Report already exists: {args.output}")
    if (args.output.resolve() in {x.resolve() for x in
            (args.test_data, args.baseline, args.tuned, args.tokenizer)}):
        raise ValueError("Output cannot overwrite inputs")
    if args.max_new_tokens < 1 or args.repetition_penalty <= 0:
        raise ValueError("Invalid generation controls")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    models = {}
    for name, checkpoint in (("baseline", args.baseline), ("tuned", args.tuned)):
        model, _ = load_model_v096(checkpoint, device, expected="rope")
        if model.vocab_size != tokenizer.vocab_size:
            raise ValueError("Tokenizer mismatch")
        models[name] = model.eval()
    context = min(m.context_length for m in models.values())
    source_rows = load_dialogue_rows(args.test_data, 0, args.seed)
    cases, stats = assemble_cases(
        source_rows, tokenizer, widths=args.widths, cases_per_width=args.cases_per_width,
        context_length=context, max_unk_fraction=args.max_unk_fraction, seed=args.seed)
    outputs = []
    for row in cases:
        expected = row["answer"]
        prompt_ids = tokenizer.encode(compact_prompt(row), add_bos=True)
        answer_ids = tokenizer.encode(expected)
        entry = {
            "case_id": row["id"],
            "source_line_sha256": hashlib.sha256(
                row["conversation_id"].encode("utf-8")).hexdigest()[:16],
            "length_chars": row["length_chars"],
            "reference_tokens": len(answer_ids),
        }
        if args.include_text:
            entry["prompt"] = compact_prompt(row)
            entry["expected"] = expected
        for name, model in models.items():
            first = first_token_diagnostic(model, prompt_ids, answer_ids)
            with torch.inference_mode():
                tokens = model.generate(
                    prompt_ids,
                    max_new_tokens=max(args.max_new_tokens, len(answer_ids) + 8),
                    eos_id=tokenizer.eos_id, temperature=0,
                    top_k=None, repetition_penalty=args.repetition_penalty)
            predicted_ids = tokens[len(prompt_ids):]
            eos = bool(predicted_ids and predicted_ids[-1] == tokenizer.eos_id)
            predicted = tokenizer.decode(predicted_ids, skip_special_tokens=True)
            metrics = score_copy(expected, predicted, eos=eos)
            metrics["first_token_correct"] = first["correct"]
            metrics["first_token_rank"] = first["target_rank"]
            if args.include_text:
                metrics["generated"] = predicted
            entry[name] = metrics
        outputs.append(entry)
    report = {
        "version": "v0.10.1",
        "frozen_models": {"baseline": str(args.baseline), "tuned": str(args.tuned)},
        "source_file": str(args.test_data),
        "total_cases": len(cases),
        "widths_chars": args.widths,
        "selection": stats,
        "evaluation": {name: summarize(outputs, name, args.widths) for name in models},
        "cases": outputs,
        "text_included": bool(args.include_text),
        "qualification": (
            "The exact target is present in the input prompt (oracle evidence). "
            "This is strictly a verbatim copying diagnostic, NOT retrieval, semantic "
            "understanding, or grounded answer accuracy. Source lines come from the "
            "Nagato TEST partition and never from train examples. Repetition penalty "
            "and EOS may affect exact matching. Report omits source text by default."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({
        "total_cases": len(cases), "selection": stats,
        "evaluation": report["evaluation"], "report": str(args.output),
    }, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Frozen RoPE evidence copy diagnostic")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--baseline", type=Path, default=Path(
        "model/model-rag-unified-nagato-v0.9.9.pt"))
    p.add_argument("--tuned", type=Path, default=Path(
        "model/model-rag-binding-v0.10.0.pt"))
    p.add_argument("--test-data", type=Path, default=Path(
        "data/dialogue/unified_v099/nagato_test_v099.jsonl"))
    p.add_argument("--output", type=Path, default=Path(
        "results/evidence_copy_diagnostics_v101.json"))
    p.add_argument("--widths", nargs="+", type=int, default=[10, 30, 50])
    p.add_argument("--cases-per-width", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--max-unk-fraction", type=float, default=.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    evaluate(args)


if __name__ == "__main__":
    main()
