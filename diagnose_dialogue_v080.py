"""LLM_RAG v0.8.0: decoding-collapse diagnostics for LLM_GPU models.

Compares identical topic prompts for the source and dialogue SFT checkpoints.
Logs stepwise probabilities, EOS rank, repetition and multiple decode policies.
Does NOT audit knowledge, validate claims, or change model parameters.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

# Deliberately not the train/val topic-only prompts of v0.7. These are
# diagnostic probes, not a formally independent holdout of the source corpus.
PROBES = (
    ("definition", "宇宙とは何ですか？"),
    ("opinion", "量子力学についてどう考えますか？"),
    ("preference", "読書は好きですか？"),
    ("identity", "あなたの名前は？"),
    ("followup", "時間についてもう少し詳しく。"),
    ("unknown", "スファレライトについて教えて。"),
)
POLICIES = ("greedy", "penalty", "sample")


def compact_prompt(question: str) -> str:
    return "問:" + question + "\n答:"


def repetition_stats(token_ids: list[int], text: str) -> dict:
    if not token_ids:
        return {"tokens": 0, "unique_token_ratio": 0.0,
                "adjacent_repeat_ratio": 0.0, "max_same_run": 0,
                "repeat_bigram_ratio": 0.0, "punctuation_ratio": 0.0}
    adjacent = sum(a == b for a, b in zip(token_ids, token_ids[1:]))
    run = maximum = 1
    for a, b in zip(token_ids, token_ids[1:]):
        run = run + 1 if a == b else 1
        maximum = max(maximum, run)
    bigrams = list(zip(token_ids, token_ids[1:]))
    return {
        "tokens": len(token_ids),
        "unique_token_ratio": len(set(token_ids)) / len(token_ids),
        "adjacent_repeat_ratio": adjacent / max(1, len(token_ids) - 1),
        "max_same_run": maximum,
        "repeat_bigram_ratio": (len(bigrams) - len(set(bigrams))) / max(1, len(bigrams)),
        "punctuation_ratio": sum(c in "、。，．,.!?！？" for c in text) / max(1, len(text)),
    }


def policy_parameters(policy: str) -> dict:
    params = {
        "greedy": {"temperature": 0.0, "top_k": None, "penalty": 1.0},
        "penalty": {"temperature": 0.0, "top_k": None, "penalty": 1.3},
        "sample": {"temperature": 0.75, "top_k": 20, "penalty": 1.2},
    }
    if policy not in params:
        raise ValueError(f"unsupported decoding policy: {policy}")
    return params[policy]


def apply_repetition_penalty(logits, ids: list[int], penalty: float):
    if penalty == 1.0:
        return logits
    adjusted = logits.clone()
    for token in set(ids):
        if 0 <= token < adjusted.numel():
            adjusted[token] = adjusted[token] / penalty if adjusted[token] >= 0 else adjusted[token] * penalty
    return adjusted


def generate_trace(model, tokenizer, question: str, policy: str,
                   max_new_tokens: int = 64, seed: int = 42) -> dict:
    """Reimplement model.generate decoding to record *pre-policy* EOS stats.

    Supports v0.3 model forward API. Preserve model weights and tokenizer.
    """
    import torch

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    cfg = policy_parameters(policy)
    prompt = compact_prompt(question)
    ids = tokenizer.encode(prompt, add_bos=True)
    if len(ids) >= model.context_length:
        return {"status": "PROMPT_TOO_LONG", "prompt": prompt, "context": model.context_length}
    prompt_unk = ids.count(tokenizer.unk_id)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.eval()
    device = next(model.parameters()).device
    history = ids[:]
    new_tokens = []
    trace = []
    stopped_eos = False
    with torch.no_grad():
        for step in range(max_new_tokens):
            x = torch.tensor([history[-model.context_length:]], device=device, dtype=torch.long)
            raw_logits = model(x)[0, -1, :].float()
            raw_probs = torch.softmax(raw_logits, dim=-1)
            raw_eos_prob = float(raw_probs[tokenizer.eos_id].item())
            # 1 means highest-ranked EOS
            raw_eos_rank = int((raw_logits > raw_logits[tokenizer.eos_id]).sum().item()) + 1
            logits = apply_repetition_penalty(raw_logits, history, cfg["penalty"])
            if cfg["temperature"] <= 0:
                chosen = int(torch.argmax(logits).item())
                adjusted_prob = float(torch.softmax(logits, dim=-1)[chosen].item())
            else:
                scaled = logits / cfg["temperature"]
                if cfg["top_k"] and cfg["top_k"] < len(scaled):
                    values, indices = torch.topk(scaled, cfg["top_k"])
                    probs = torch.softmax(values, dim=-1)
                    offset = int(torch.multinomial(probs, 1).item())
                    chosen = int(indices[offset].item())
                    adjusted_prob = float(probs[offset].item())
                else:
                    probs = torch.softmax(scaled, dim=-1)
                    chosen = int(torch.multinomial(probs, 1).item())
                    adjusted_prob = float(probs[chosen].item())
            char = tokenizer.decode([chosen], skip_special_tokens=False)
            trace.append({
                "step": step + 1,
                "token_id": chosen,
                "token": char,
                "raw_eos_probability": round(raw_eos_prob, 8),
                "raw_eos_rank": raw_eos_rank,
                "chosen_policy_probability": round(adjusted_prob, 6),
                "raw_top1_probability": round(float(raw_probs.max().item()), 6),
            })
            history.append(chosen)
            if chosen == tokenizer.eos_id:
                stopped_eos = True
                break
            new_tokens.append(chosen)
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    stats = repetition_stats(new_tokens, response)
    return {
        "status": "OK", "prompt": prompt, "prompt_unk_count": prompt_unk,
        "context": model.context_length, "policy": policy, "text": response,
        "eos_generated": stopped_eos, "generation_length": len(new_tokens),
        "repetition": stats,
        "mean_raw_eos_probability": (sum(t["raw_eos_probability"] for t in trace) / len(trace)) if trace else None,
        "min_raw_eos_rank": min((t["raw_eos_rank"] for t in trace), default=None),
        "trace": trace,
    }


def aggregate(rows: list[dict]) -> dict:
    grouped = {}
    for row in rows:
        key = row["model"] + "/" + row["policy"]
        result = row["result"]
        if result["status"] != "OK":
            continue
        grouped.setdefault(key, []).append(result)
    report = {}
    for key, group in grouped.items():
        report[key] = {
            "cases": len(group),
            "eos_count": sum(r["eos_generated"] for r in group),
            "mean_adjacent_repeat_ratio": round(sum(r["repetition"]["adjacent_repeat_ratio"] for r in group) / len(group), 4),
            "mean_unique_token_ratio": round(sum(r["repetition"]["unique_token_ratio"] for r in group) / len(group), 4),
            "mean_punctuation_ratio": round(sum(r["repetition"]["punctuation_ratio"] for r in group) / len(group), 4),
            "mean_generation_length": round(sum(r["generation_length"] for r in group) / len(group), 1),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="Diagnose repetition collapse in LLM_GPU v0.7")
    parser.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    parser.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    parser.add_argument("--baseline", type=Path, default=Path("model/model-rag-nagato-v0.2.pt"))
    parser.add_argument("--tuned", type=Path, default=Path("model/model-rag-dialogue-v0.7.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/dialogue_diagnostics_v080.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.output}. Use --overwrite.")
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be >= 1")
    import torch
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from model import LanguageModel
    from tokenizer import Tokenizer

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    models = {}
    for name, filename in (("baseline", args.baseline), ("tuned", args.tuned)):
        model, _ = LanguageModel.load_checkpoint(str(filename), device=device)
        if model.vocab_size != tokenizer.vocab_size:
            raise ValueError(f"Vocabulary mismatch: {name}")
        models[name] = model
    rows = []
    for name, model in models.items():
        for category, question in PROBES:
            for policy in POLICIES:
                result = generate_trace(model, tokenizer, question, policy,
                                        max_new_tokens=args.max_new_tokens, seed=args.seed)
                rows.append({"model": name, "category": category, "question": question,
                             "policy": policy, "result": result})
                print(f"[{name}/{policy}/{category}] eos={result.get('eos_generated')} "
                      f"repeat={result.get('repetition', {}).get('adjacent_repeat_ratio')} "
                      f"answer={result.get('text', '')[:90]!r}")
    report = {
        "version": "v0.8.0",
        "device": str(device),
        "checkpoints": {"baseline": str(args.baseline), "tuned": str(args.tuned)},
        "sampled_probes": [{"category": c, "question": q} for c, q in PROBES],
        "aggregate": aggregate(rows), "cases": rows,
        "limitations": "Questions are fixed new prompts, but the earlier Nagato source corpus overlaps training. This is diagnostic, not a fully independent factual or conversational benchmark.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Summary:", json.dumps(report["aggregate"], ensure_ascii=False))
    print("Report:", args.output)


if __name__ == "__main__":
    main()
