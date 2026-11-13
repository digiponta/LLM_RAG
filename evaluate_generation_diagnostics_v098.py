"""LLM_RAG v0.9.8 — first-token and gold-prefix continuation diagnostics.

This is an analysis tool, not another fine-tuning procedure. It compares the
same held-out examples for two frozen models. Teacher forcing, free generation
and gold-prefix-conditioned generation are scored separately.

No private question or reference text is written unless --include-text is set.
No writes except the requested local JSON result; no network calls.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from prepare_real_persona_v081 import compact_prompt
from train_real_dialogue_v082 import load_dialogue_rows, encode_dialogue


def first_token_diagnostic(model, prompt_ids: list[int], answer_ids: list[int]):
    """Rank/probability of the exact reference first token at prompt boundary."""
    import torch
    if not answer_ids:
        raise ValueError("Empty reference answer")
    if len(prompt_ids) >= model.context_length:
        raise ValueError("Prompt exceeds context")
    with torch.inference_mode():
        device = next(model.parameters()).device
        tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits = model(tensor)[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        target = answer_ids[0]
        rank = int((logits > logits[target]).sum().item()) + 1
        selected = int(logits.argmax().item())
        return {
            "target_token_id": int(target),
            "predicted_token_id": selected,
            "correct": selected == target,
            "target_rank": rank,
            "target_probability": float(log_probs[target].exp().item()),
            "target_nll": float(-log_probs[target].item()),
        }


def prefix_continuation_diagnostic(model, tokenizer, prompt_ids: list[int],
                                    answer_ids: list[int], prefix_tokens: int,
                                    *, generation_limit: int = 80):
    """Generate remaining answer after providing N *gold* answer tokens.

    Prefix must be strictly shorter than the reference answer; generated
    text is evaluated against the reference SUFFIX only, not the full answer.
    """
    if not 1 <= prefix_tokens < len(answer_ids):
        raise ValueError("Prefix length must be in [1, answer length)")
    if generation_limit < 1:
        raise ValueError("Invalid generation limit")
    priming = prompt_ids + answer_ids[:prefix_tokens]
    if len(priming) >= model.context_length:
        raise ValueError("Gold prefix exceeds model context")
    import torch
    with torch.inference_mode():
        generated = model.generate(priming, max_new_tokens=generation_limit,
                                   eos_id=tokenizer.eos_id, temperature=0,
                                   top_k=None, repetition_penalty=1.3)
    continuation = generated[len(priming):]
    if continuation and continuation[-1] == tokenizer.eos_id:
        continuation = continuation[:-1]
    expected = answer_ids[prefix_tokens:]
    common = 0
    for pred, gold in zip(continuation, expected):
        if pred != gold:
            break
        common += 1
    return {
        "prefix_tokens": prefix_tokens,
        "remaining_reference_tokens": len(expected),
        "continuation_exact": continuation == expected,
        "matched_leading_tokens": common,
        "leading_token_coverage": common / len(expected),
        "generated_tokens": len(continuation),
        "expected_tokens": len(expected),
        "generated_suffix": tokenizer.decode(continuation, skip_special_tokens=True),
        "expected_suffix": tokenizer.decode(expected, skip_special_tokens=True),
        "reference_prefix": tokenizer.decode(answer_ids[:prefix_tokens],
                                              skip_special_tokens=True),
    }


def summarize(results: list[dict]) -> dict:
    if not results:
        return {"count": 0}
    first = [x["first_token"] for x in results]
    output = {
        "count": len(results),
        "first_token_accuracy": sum(x["correct"] for x in first) / len(first),
        "mean_target_rank": sum(x["target_rank"] for x in first) / len(first),
        "mean_target_nll": sum(x["target_nll"] for x in first) / len(first),
        "top5_target_fraction": sum(x["target_rank"] <= 5 for x in first) / len(first),
        "prefix": {},
    }
    widths = sorted({key for row in results for key in row["prefix"]},
                    key=lambda n: int(n))
    for key in widths:
        valid = [row["prefix"][key] for row in results if key in row["prefix"]]
        output["prefix"][key] = {
            "cases": len(valid),
            "continuation_exact_fraction": sum(x["continuation_exact"] for x in valid) / len(valid),
            "mean_leading_token_coverage": sum(x["leading_token_coverage"] for x in valid) / len(valid),
        }
    return output


def evaluate(args):
    import torch
    from tokenizer import Tokenizer
    from mha_rope_v096 import load_model_v096
    from intent_model_v097 import load_intent_model
    if args.sample < 1 or args.max_new_tokens < 1 or not args.prefix_tokens:
        raise ValueError("Invalid evaluation options")
    if any(k < 1 for k in args.prefix_tokens) or len(set(args.prefix_tokens)) != len(args.prefix_tokens):
        raise ValueError("Prefix sizes must be positive and distinct")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Existing report: {args.output}")
    input_paths = (args.baseline, args.tuned, args.test_data, args.tokenizer)
    if args.output.resolve() in {p.resolve() for p in input_paths}:
        raise ValueError("Output must not overwrite any input")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    tok = Tokenizer.load(str(args.tokenizer))
    loaded = {}
    for key, path, use_intent in (
        ("baseline", args.baseline, args.baseline_intent),
        ("tuned", args.tuned, args.tuned_intent),
    ):
        if use_intent:
            model, _ = load_intent_model(path, device)
        else:
            model, _ = load_model_v096(path, device, expected="rope")
        if model.vocab_size != tok.vocab_size:
            raise ValueError(f"{key}: tokenizer vocab mismatch")
        loaded[key] = model.eval()
    context = min(model.context_length for model in loaded.values())
    rows = load_dialogue_rows(args.test_data, args.sample, args.seed)
    stats = Counter()
    cases = []
    for row in rows:
        encoded, reason = encode_dialogue(row, tok, context, args.max_unk_fraction)
        stats[reason] += 1
        if encoded is None:
            continue
        prompt_ids = tok.encode(compact_prompt(row), add_bos=True)
        answer_ids = tok.encode(row["answer"])
        entry = {"case_id": row["id"], "reference_answer_tokens": len(answer_ids)}
        if args.include_text:
            entry["question"] = row["question"]
            entry["reference_answer"] = row["answer"]
        for model_name, model in loaded.items():
            info = {
                "first_token": first_token_diagnostic(model, prompt_ids, answer_ids),
                "prefix": {},
            }
            for k in args.prefix_tokens:
                if k >= len(answer_ids):
                    continue
                diagnostic = prefix_continuation_diagnostic(
                    model, tok, prompt_ids, answer_ids, k,
                    generation_limit=args.max_new_tokens
                )
                if not args.include_text:
                    for name in ("generated_suffix", "expected_suffix", "reference_prefix"):
                        diagnostic.pop(name)
                info["prefix"][str(k)] = diagnostic
            entry[model_name] = info
        cases.append(entry)
    if not cases:
        raise ValueError("No usable held-out rows")
    report = {
        "version": "v0.9.8",
        "baseline_checkpoint": str(args.baseline),
        "tuned_checkpoint": str(args.tuned),
        "test_rows_selected": len(rows),
        "test_rows_usable": len(cases),
        "filter_counts": dict(stats),
        "prefix_tokens": args.prefix_tokens,
        "max_new_tokens": args.max_new_tokens,
        "models": {key: summarize([row[key] for row in cases])
                   for key in loaded},
        "cases": cases,
        "text_included": bool(args.include_text),
        "limitations": (
            "First-token target is the first character-token of the full reference answer, "
            "not an intent or a semantic label. Gold-prefix evaluation gives privileged "
            "answer text; it is diagnostic, not deployable generation. "
            "A small sample and teacher-forced improvements cannot establish knowledge, "
            "question relevance or factuality. Private text is omitted by default."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({
        "test_rows_selected": len(rows),
        "test_rows_usable": len(cases),
        "models": report["models"],
        "report": str(args.output),
    }, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="First-token vs gold-prefix decoding diagnostics")
    parser.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("../LLM_GPU/model/tokenizer.json"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("model/model-rag-rope-medium-v0.9.5.pt"))
    parser.add_argument("--tuned", type=Path,
                        default=Path("model/model-rag-concepts-v0.9.7.pt"))
    parser.add_argument("--test-data", type=Path,
                        default=Path("data/dialogue/concepts_v097/concept_test_v097.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/concept_generation_diagnostics_v098.json"))
    parser.add_argument("--baseline-intent", action="store_true")
    parser.add_argument("--tuned-intent", action="store_true")
    parser.add_argument("--prefix-tokens", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--sample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-unk-fraction", type=float, default=.10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    evaluate(args)


if __name__ == "__main__":
    main()
