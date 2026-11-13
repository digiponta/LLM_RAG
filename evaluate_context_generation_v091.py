"""LLM_RAG v0.9.1: review generated responses WITH their authentic context.

This is a read-only diagnostic; no changes to checkpoint or dataset.
Private dialogue text appears ONLY in an explicitly requested local report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from prepare_real_persona_v081 import compact_prompt
from train_real_dialogue_v082 import load_dialogue_rows, encode_dialogue
from evaluate_context_swap_v088 import answer_nll


def case_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def plain_variant(row: dict) -> dict:
    return {**row, "context": ""}


def generate_response(model, tokenizer, row: dict, max_new_tokens: int) -> dict:
    prompt = tokenizer.encode(compact_prompt(row), add_bos=True)
    # The existing generator slides its attention window as tokens accumulate.
    if len(prompt) >= model.context_length:
        return {"status": "PROMPT_TOO_LONG"}
    ids = model.generate(prompt, max_new_tokens=max_new_tokens,
                         eos_id=tokenizer.eos_id, temperature=0.0,
                         top_k=None, repetition_penalty=1.3)
    generated = ids[len(prompt):]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "status": "OK",
        "text": response,
        "response_hash": case_hash(response),
        "chars": len(response),
        "eos": bool(generated and generated[-1] == tokenizer.eos_id),
        "exact_match": response == row["answer"],
    }


def summarize_cases(cases: list[dict]) -> dict:
    if not cases:
        return {"cases": 0}
    valid = [c for c in cases if c["status"] == "OK"]
    if not valid:
        return {"cases": len(cases), "valid": 0}
    return {
        "cases": len(cases),
        "valid": len(valid),
        "eos_rate": sum(c["eos"] for c in valid) / len(valid),
        "exact_match_rate": sum(c["exact_match"] for c in valid) / len(valid),
        "mean_chars": sum(c["chars"] for c in valid) / len(valid),
        "unique_response_ratio": len({c["response_hash"] for c in valid}) / len(valid),
    }


def compare_row(model, tokenizer, row: dict, device, max_new_tokens: int) -> dict:
    """Make the SAME question and reference answer under two input conditions."""
    contextual, reason_c = encode_dialogue(row, tokenizer, model.context_length)
    plain_row = plain_variant(row)
    plain, reason_p = encode_dialogue(plain_row, tokenizer, model.context_length)
    if contextual is None or plain is None:
        return {"status": "SKIPPED", "context_reason": reason_c, "plain_reason": reason_p}
    c = generate_response(model, tokenizer, row, max_new_tokens)
    p = generate_response(model, tokenizer, plain_row, max_new_tokens)
    if c["status"] != "OK" or p["status"] != "OK":
        return {"status": "SKIPPED", "context_reason": c["status"], "plain_reason": p["status"]}
    return {
        "status": "OK", "context_generation": c, "plain_generation": p,
        "context_answer_nll": answer_nll(model, contextual, device),
        "plain_answer_nll": answer_nll(model, plain, device),
        "context_nll_advantage": answer_nll(model, plain, device) - answer_nll(model, contextual, device),
        "response_changed": c["text"] != p["text"],
    }


def display_case(row: dict, result: dict) -> dict:
    """Local review record with original historical turns and full generations."""
    return {
        "history": row["context"].split("／") if row["context"] else [],
        "question": row["question"],
        "reference_answer": row["answer"],
        "question_speaker": row.get("question_speaker"),
        "answer_speaker": row.get("answer_speaker"),
        "with_context": result["context_generation"]["text"],
        "without_context": result["plain_generation"]["text"],
        "context_answer_nll": result["context_answer_nll"],
        "plain_answer_nll": result["plain_answer_nll"],
        "context_nll_advantage": result["context_nll_advantage"],
        "response_changed": result["response_changed"],
        "note": "History is shown in chronological order; per-turn speaker labels are not recoverable from this derived context string.",
    }


def run(args) -> dict:
    if args.sample < 1 or args.max_new_tokens < 1:
        raise ValueError("sample and max-new-tokens must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}")
    if args.output.resolve() in {args.checkpoint.resolve(), args.test_data.resolve(), args.tokenizer.resolve()}:
        raise ValueError("Output cannot overwrite inputs")
    if args.include_text and not args.output.name.endswith("_text_v091.json"):
        raise ValueError("Text-bearing report must be named *_text_v091.json")

    import torch
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from tokenizer import Tokenizer
    from rope_model_v093 import load_configured_model
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model, _ = load_configured_model(args.checkpoint, device, rope=args.rope)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError("Tokenizer mismatch")
    model.eval()

    rows = load_dialogue_rows(args.test_data, args.sample, args.seed)
    results = []
    for row in rows:
        outcome = compare_row(model, tokenizer, row, device, args.max_new_tokens)
        if outcome["status"] != "OK":
            continue
        item = {
            "case_hash": case_hash(row["id"]),
            "context_answer_nll": outcome["context_answer_nll"],
            "plain_answer_nll": outcome["plain_answer_nll"],
            "context_nll_advantage": outcome["context_nll_advantage"],
            "response_changed": outcome["response_changed"],
            "context_generation": {k: v for k, v in outcome["context_generation"].items() if k != "text"},
            "plain_generation": {k: v for k, v in outcome["plain_generation"].items() if k != "text"},
        }
        if args.include_text:
            item["review"] = display_case(row, outcome)
        results.append(item)
    if not results:
        raise ValueError("No usable matched question/answer/context cases")
    margins = [r["context_nll_advantage"] for r in results]
    summary = {
        "selected": len(rows),
        "usable": len(results),
        "response_change_rate": sum(r["response_changed"] for r in results) / len(results),
        "mean_context_nll_advantage": sum(margins) / len(margins),
        "positive_context_nll_advantage_rate": sum(m > 0 for m in margins) / len(margins),
        "context_generation": summarize_cases([r["context_generation"] for r in results]),
        "plain_generation": summarize_cases([r["plain_generation"] for r in results]),
    }
    report = {
        "version": "v0.9.1",
        "checkpoint": str(args.checkpoint), "test_data": str(args.test_data),
        "context_length": model.context_length, "summary": summary, "cases": results,
        "contains_raw_dialogue_text": args.include_text,
        "limitations": (
            "Only one model is compared under two input formats, not matched context versus "
            "a topic-matched counterfactual. More changed outputs do not mean better responses. "
            "The original context joins turns using ／ but does not preserve speaker IDs per turn. "
            "No automatic semantic/grammatical quality scores. Dialogue-file splits may overlap "
            "by participant/topic; prompts are short-turn selected. Text-bearing reports "
            "contain potentially private Real Persona Chat material and must remain local."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Report:", args.output)
    return report


def main():
    p = argparse.ArgumentParser(description="v0.9.1 context-visible generation evaluation")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--checkpoint", type=Path, default=Path("model/model-rag-context256-history-v0.9.0.pt"))
    p.add_argument("--test-data", type=Path, default=Path("data/dialogue/real_persona_context256_v090/paired_context_test_v087.jsonl"))
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--output", type=Path, default=Path("results/context_generation_v091.json"))
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--rope", action="store_true", help="Load RoPE checkpoint explicitly")
    p.add_argument("--overwrite", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
