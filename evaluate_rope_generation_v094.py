"""v0.9.4 — paired, history-visible generation comparison: plain Attention vs RoPE.

Frozen models; same held-out dialogue IDs, prompt strings, reference answers,
greedy decoding and answer-token NLL. No automatic semantic quality claims.
Raw private text requires --include-text and a special local filename.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluate_context_generation_v091 import case_hash, generate_response, summarize_cases
from evaluate_context_swap_v088 import answer_nll
from train_real_dialogue_v082 import encode_dialogue, load_dialogue_rows


def build_review(row: dict, plain: dict, rope: dict, nll_plain: float,
                 nll_rope: float) -> dict:
    return {
        "history": row.get("context", "").split("／") if row.get("context") else [],
        "question": row["question"],
        "reference_answer": row["answer"],
        "legacy_response": plain["text"],
        "rope_response": rope["text"],
        "legacy_nll": nll_plain,
        "rope_nll": nll_rope,
        "nll_gain": nll_plain - nll_rope,
        "review_fields": {
            "legacy": {"natural_japanese": None, "question_relevance": None,
                       "context_reference": None, "topic_consistency": None},
            "rope": {"natural_japanese": None, "question_relevance": None,
                     "context_reference": None, "topic_consistency": None},
        },
        "rating_scale": "0=incorrect/unintelligible; 1=partial; 2=adequate. Null=not reviewed.",
        "note": "History chronology is preserved; derived history lacks per-turn speaker identity.",
    }


def summarize(records: list[dict], selected: int) -> dict:
    if not records:
        return {"selected": selected, "usable": 0}
    gains = [r["nll_gain"] for r in records]
    return {
        "selected": selected, "usable": len(records),
        "mean_example_nll_gain": sum(gains) / len(gains),
        "rope_lower_nll_rate": sum(g > 0 for g in gains) / len(gains),
        "response_change_rate": sum(r["response_changed"] for r in records) / len(records),
        "legacy_generation": summarize_cases([r["legacy_generation"] for r in records]),
        "rope_generation": summarize_cases([r["rope_generation"] for r in records]),
    }


def verify_paths(args) -> None:
    if args.sample < 1 or args.max_new_tokens < 1:
        raise ValueError("sample and max-new-tokens must be positive")
    if args.include_text and not args.output.name.endswith("_text_v094.json"):
        raise ValueError("Raw-text report filename must end in _text_v094.json")
    inputs = {x.resolve() for x in (args.baseline, args.rope_checkpoint,
                                   args.test_data, args.tokenizer)}
    if args.output.resolve() in inputs:
        raise ValueError("Output cannot overwrite an input")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")


def run(args) -> dict:
    verify_paths(args)
    import torch
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from tokenizer import Tokenizer
    from mha_rope_v096 import load_model_v096

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    tok = Tokenizer.load(str(args.tokenizer))
    legacy, _ = load_model_v096(args.baseline, device, expected="legacy")
    use_mha = getattr(args, "mha", False)
    rope, _ = load_model_v096(
        args.rope_checkpoint, device, expected="mha" if use_mha else "rope",
        mha_heads=getattr(args, "mha_heads", 4))
    if (legacy.vocab_size != rope.vocab_size or legacy.vocab_size != tok.vocab_size or
        legacy.context_length != rope.context_length):
        raise ValueError("Model vocab/context mismatch")
    legacy.eval()
    rope.eval()

    rows = load_dialogue_rows(args.test_data, args.sample, args.seed)
    out = []
    excluded = {}
    for row in rows:
        pair, reason = encode_dialogue(row, tok, legacy.context_length)
        if pair is None:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        generated_legacy = generate_response(legacy, tok, row, args.max_new_tokens)
        generated_rope = generate_response(rope, tok, row, args.max_new_tokens)
        if generated_legacy["status"] != "OK" or generated_rope["status"] != "OK":
            excluded["generation_prompt_too_long"] = excluded.get("generation_prompt_too_long", 0) + 1
            continue
        nll_legacy = answer_nll(legacy, pair, device)
        nll_rope = answer_nll(rope, pair, device)
        record = {
            "case_hash": case_hash(row["id"]),
            "legacy_nll": nll_legacy,
            "rope_nll": nll_rope,
            "nll_gain": nll_legacy - nll_rope,
            "response_changed": generated_legacy["text"] != generated_rope["text"],
            "legacy_generation": {k: v for k, v in generated_legacy.items() if k != "text"},
            "rope_generation": {k: v for k, v in generated_rope.items() if k != "text"},
        }
        if args.include_text:
            record["review"] = build_review(
                row, generated_legacy, generated_rope, nll_legacy, nll_rope)
        out.append(record)
    if not out:
        raise ValueError("No valid held-out examples")
    summary = summarize(out, len(rows))
    report = {
        "version": "v0.9.4",
        "legacy_checkpoint": str(args.baseline),
        "rope_checkpoint": str(args.rope_checkpoint),
        "data": str(args.test_data),
        "device": str(device),
        "seed": args.seed,
        "context_length": legacy.context_length,
        "contains_raw_dialogue_text": bool(args.include_text),
        "summary": summary,
        "excluded": excluded,
        "cases": out,
        "limitations": (
            "NLL is per-example answer-token mean, not corpus token-weighted NLL; "
            "compare to v0.9.3 aggregate only by re-running both here. "
            "Response change, length, diversity and EOS are NOT quality judgments. "
            "This script does not assign semantic, grammatical or factual scores. "
            "Only a small sampled dialogue-file-separated set is reviewed; original "
            "participants/topics may overlap between splits. "
            "Raw text is private; text-bearing reports must stay local."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "excluded": excluded,
                      "report": str(args.output)}, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Compare frozen Legacy and RoPE generation on identical dialogs")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--baseline", type=Path, default=Path("model/model-rag-legacy-paired-v0.9.2.pt"))
    p.add_argument("--rope-checkpoint", type=Path, default=Path("model/model-rag-rope-legacy-v0.9.3.pt"))
    p.add_argument("--test-data", type=Path, default=Path("data/dialogue/real_persona_structured_v092/paired_legacy_test_v092.jsonl"))
    p.add_argument("--sample", type=int, default=30)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--mha", action="store_true",
                   help="Compare baseline to a multi-head RoPE checkpoint instead of single-head")
    p.add_argument("--mha-heads", type=int, default=4)
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--output", type=Path, default=Path("results/rope_generation_v094.json"))
    p.add_argument("--overwrite", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
