"""LLM_RAG v0.8.6: controlled decoding diagnostics on REAL held-out dialogue.

Model and tokenizer are read-only. Real conversation text is NOT present in
reports by default. Scores do not claim to measure answer correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from diagnose_dialogue_v080 import apply_repetition_penalty, repetition_stats
from prepare_real_persona_v081 import compact_prompt
from train_real_dialogue_v082 import load_dialogue_rows

POLICIES = {
    "greedy": {"temperature": 0.0, "top_k": None, "penalty": 1.0},
    "penalty": {"temperature": 0.0, "top_k": None, "penalty": 1.3},
    "sample": {"temperature": 0.8, "top_k": None, "penalty": 1.0},
    "topk": {"temperature": 0.8, "top_k": 20, "penalty": 1.0},
    "topk_penalty": {"temperature": 0.8, "top_k": 20, "penalty": 1.3},
}


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def config_for(policy: str) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    return dict(POLICIES[policy])


def generate_one(model, tokenizer, question: str, policy: str,
                 max_new_tokens: int, seed: int) -> dict:
    import torch

    cfg = config_for(policy)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    history = tokenizer.encode("問:" + question + "\n答:", add_bos=True)
    if len(history) >= model.context_length:
        return {"status": "PROMPT_TOO_LONG"}
    prompt_length = len(history)
    unk_fraction = history.count(tokenizer.unk_id) / len(history)
    if unk_fraction > 0.10:
        return {"status": "HIGH_PROMPT_UNK"}
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    generated = []
    stopped_eos = False
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            x = torch.tensor([history[-model.context_length:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1].float()
            logits = apply_repetition_penalty(logits, history, cfg["penalty"])
            if cfg["temperature"] == 0.0:
                token = int(torch.argmax(logits).item())
            else:
                logits = logits / cfg["temperature"]
                if cfg["top_k"] is not None:
                    top_values, top_ids = torch.topk(logits, min(cfg["top_k"], logits.numel()))
                    probabilities = torch.softmax(top_values, dim=-1)
                    idx = int(torch.multinomial(probabilities, 1, generator=generator).item())
                    token = int(top_ids[idx].item())
                else:
                    probabilities = torch.softmax(logits, dim=-1)
                    token = int(torch.multinomial(probabilities, 1, generator=generator).item())
            history.append(token)
            if token == tokenizer.eos_id:
                stopped_eos = True
                break
            generated.append(token)

    response = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "status": "OK",
        "response": response,
        "response_hash": hash_text(response),
        "response_chars": len(response),
        "eos": stopped_eos,
        "generated_tokens": len(generated),
        "prompt_tokens": prompt_length,
        "repetition": repetition_stats(generated, response),
    }


def summarize(cases: list[dict]) -> dict:
    """Aggregate WITHOUT source text. Intra-policy comparisons use same prompts."""
    if not cases:
        return {"cases": 0}
    valid = [r for r in cases if r["status"] == "OK"]
    if not valid:
        return {"cases": len(cases), "valid": 0}
    hashes = Counter(r["response_hash"] for r in valid)
    return {
        "cases": len(cases),
        "valid": len(valid),
        "eos_rate": sum(r["eos"] for r in valid) / len(valid),
        "empty_rate": sum(not r["response_chars"] for r in valid) / len(valid),
        "unique_response_ratio": len(hashes) / len(valid),
        "most_common_response_share": max(hashes.values()) / len(valid),
        "mean_response_chars": sum(r["response_chars"] for r in valid) / len(valid),
        "mean_adjacent_repeat_ratio": sum(r["repetition"]["adjacent_repeat_ratio"] for r in valid) / len(valid),
        "mean_repeat_bigram_ratio": sum(r["repetition"]["repeat_bigram_ratio"] for r in valid) / len(valid),
        "most_common_response_hash": hashes.most_common(1)[0][0],
    }


def run(args) -> dict:
    if args.samples < 1 or args.max_new_tokens < 1:
        raise ValueError("samples and max-new-tokens must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Existing report: {args.output}")
    guarded = [args.test_data, args.checkpoint, args.tokenizer]
    if args.baseline:
        guarded.append(args.baseline)
    if args.output.resolve() in {p.resolve() for p in guarded}:
        raise ValueError("Report must not overwrite an input")
    if args.include_text and not args.output.name.endswith("_text_v086.json"):
        raise ValueError("For text-bearing output use a distinct *_text_v086.json filename")

    import torch
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from tokenizer import Tokenizer
    from model import LanguageModel
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    rows = load_dialogue_rows(args.test_data, args.samples, args.seed)
    models = {"full_v084": args.checkpoint}
    if args.baseline:
        models["pilot_v082"] = args.baseline
    results = {}
    examples = []
    for model_name, filename in models.items():
        model, _ = LanguageModel.load_checkpoint(str(filename), device=device)
        if tokenizer.vocab_size != model.vocab_size:
            raise ValueError("Tokenizer/checkpoint mismatch")
        for policy in POLICIES:
            runs = []
            for i, row in enumerate(rows):
                # Exact same seeded prompt set for each policy/model.
                out = generate_one(model, tokenizer, row["question"], policy,
                                   args.max_new_tokens, args.seed + i * 17)
                runs.append(out)
                record = {"id_hash": hash_text(row["id"]), "model": model_name,
                          "policy": policy, "status": out["status"]}
                if out["status"] == "OK":
                    record.update({key: out[key] for key in (
                        "response_hash", "response_chars", "eos", "generated_tokens",
                        "prompt_tokens", "repetition",
                    )})
                    if args.include_text:
                        record["question"] = row["question"]
                        record["reference"] = row["answer"]
                        record["response"] = out["response"]
                examples.append(record)
            results[f"{model_name}/{policy}"] = summarize(runs)
            print(f"[{model_name}/{policy}] {json.dumps(results[f'{model_name}/{policy}'], ensure_ascii=False)}")
    report = {
        "version": "v0.8.6",
        "device": str(device),
        "source_test": str(args.test_data),
        "test_sampled": len(rows),
        "parameters": {"seed": args.seed, "max_new_tokens": args.max_new_tokens,
                       "policies": POLICIES},
        "aggregate": results,
        "cases": examples,
        "contains_raw_dialogue_text": args.include_text,
        "limitations": (
            "All results are decoding diagnostics, not semantic correctness scores. "
            "High diversity can mean nonsense, and high EOS rates can mean early stopping. "
            "Real Persona Chat test is split by dialogue file, not speaker/near duplicate. "
            "Responses are generated autoregressively with sliding context once full."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Report:", args.output)
    return report


def main():
    p = argparse.ArgumentParser(description="v0.8.6 real-dialogue generation diagnostics")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--checkpoint", type=Path, default=Path("model/model-rag-real-dialogue-full-v0.8.4.pt"))
    p.add_argument("--baseline", type=Path, default=None)
    p.add_argument("--test-data", type=Path, default=Path("data/dialogue/real_persona_v081/real_persona_test_v081.jsonl"))
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--output", type=Path, default=Path("results/generation_diagnostics_v086.json"))
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
