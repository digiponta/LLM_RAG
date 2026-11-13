"""LLM_RAG v0.8.8: held-out matched vs counterfactual-context answer NLL.

No training, private text excluded by default. Swap only the context, preserve
question and answer. Never present this as a full semantic-understanding test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from train_real_dialogue_v082 import load_dialogue_rows, encode_dialogue


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def choose_donor(target: dict, pool: list[dict], tokenizer, context_limit: int,
                 rng: random.Random, strategy: str, length_tolerance: int = 6):
    """Find an eligible donor from a different conversation, with valid encoding."""
    if strategy not in ("random", "length_matched"):
        raise ValueError("Unknown swap strategy")
    original_context_tokens = len(tokenizer.encode(target["context"]))
    candidates = []
    for donor in pool:
        if (donor["id"] == target["id"] or
            donor["conversation_id"] == target["conversation_id"] or
            donor["context"] == target["context"] or
            donor["question"] == target["question"]):
            continue
        n_tokens = len(tokenizer.encode(donor["context"]))
        if strategy == "length_matched" and abs(n_tokens - original_context_tokens) > length_tolerance:
            continue
        swapped = dict(target, context=donor["context"])
        pair, reason = encode_dialogue(swapped, tokenizer, context_limit)
        if pair is not None:
            candidates.append((donor, swapped, abs(n_tokens - original_context_tokens)))
    if not candidates:
        return None
    if strategy == "length_matched":
        # Randomize within nearest-length candidates, not order of input JSONL.
        best_delta = min(candidate[2] for candidate in candidates)
        candidates = [candidate for candidate in candidates if candidate[2] == best_delta]
    donor, swapped, distance = rng.choice(candidates)
    return {"donor": donor, "swapped": swapped, "length_delta": distance}


def answer_nll(model, pair, device) -> float:
    """Mean token-level cross entropy of the answer alone for one paired example."""
    import torch
    import torch.nn.functional as F

    inputs, labels = pair
    x = torch.tensor([inputs], device=device, dtype=torch.long)
    y = torch.tensor(labels, device=device, dtype=torch.long)
    with torch.inference_mode():
        logits = model(x)
        total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                y.reshape(-1), ignore_index=-100, reduction="sum")
    count = int((y != -100).sum().item())
    if count == 0:
        raise ValueError("No supervised answer tokens")
    return float(total.item()) / count


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"cases": 0}
    margins = [r["margin"] for r in records]
    return {
        "cases": len(records),
        "mean_nll_margin": sum(margins) / len(margins),
        "positive_margin_rate": sum(m > 0 for m in margins) / len(margins),
        "mean_absolute_margin": sum(abs(m) for m in margins) / len(margins),
        "mean_correct_context_nll": sum(r["correct_nll"] for r in records) / len(records),
        "mean_swapped_context_nll": sum(r["swapped_nll"] for r in records) / len(records),
        "mean_context_token_length_delta": sum(r["length_delta"] for r in records) / len(records),
    }


def evaluate(args):
    if args.sample < 2 or args.length_tolerance < 0:
        raise ValueError("sample must be >=2, tolerance >=0")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}")
    if args.output.resolve() in {args.test_data.resolve(), args.checkpoint.resolve(),
                                 args.tokenizer.resolve()}:
        raise ValueError("Output cannot overwrite an input")
    if args.include_text and not args.output.name.endswith("_text_v088.json"):
        raise ValueError("Text-bearing reports must use *_text_v088.json")
    import torch
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from tokenizer import Tokenizer
    from rope_model_v093 import load_configured_model

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    model, _ = load_configured_model(args.checkpoint, device, rope=args.rope)
    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError("Checkpoint/tokenizer mismatch")
    model.eval()
    rows = load_dialogue_rows(args.test_data, args.sample, args.seed)
    rows = [row for row in rows if row.get("context")]
    if len(rows) < 2:
        raise ValueError("Too few contextual cases")
    valid = []
    for row in rows:
        encoded, status = encode_dialogue(row, tokenizer, model.context_length)
        if encoded is not None:
            valid.append((row, encoded))
    if len(valid) < 2:
        raise ValueError("Too few context cases fit model")
    pool = [row for row, _ in valid]
    report = {"version": "v0.8.8", "checkpoint": str(args.checkpoint),
              "test_data": str(args.test_data), "test_rows_selected": len(rows),
              "correct_context_usable": len(valid), "device": str(device),
              "seed": args.seed, "length_tolerance": args.length_tolerance,
              "contains_raw_dialogue_text": bool(args.include_text),
              "controls": {}, "cases": []}
    rng = random.Random(args.seed)
    for strategy in ("random", "length_matched"):
        results = []
        for row, encoded in valid:
            chosen = choose_donor(row, pool, tokenizer, model.context_length,
                                  rng, strategy, args.length_tolerance)
            if chosen is None:
                continue
            swapped_pair, _ = encode_dialogue(chosen["swapped"], tokenizer, model.context_length)
            if swapped_pair is None:
                raise AssertionError("Invalid selected donor")
            correct_nll = answer_nll(model, encoded, device)
            swapped_nll = answer_nll(model, swapped_pair, device)
            record = {
                "case_hash": short_hash(row["id"]),
                "donor_hash": short_hash(chosen["donor"]["id"]),
                "strategy": strategy,
                "correct_nll": correct_nll,
                "swapped_nll": swapped_nll,
                "margin": swapped_nll - correct_nll,
                "length_delta": chosen["length_delta"],
            }
            if args.include_text:
                record.update({"question": row["question"], "answer": row["answer"],
                               "correct_context": row["context"],
                               "swapped_context": chosen["swapped"]["context"]})
            results.append(record)
        report["controls"][strategy] = {
            **summarize(results),
            "excluded_no_eligible_donor": len(valid) - len(results),
        }
        report["cases"].extend(results)
        print(f"{strategy}: " + json.dumps(report["controls"][strategy], ensure_ascii=False))
    report["limitations"] = (
        "This tests whether the original dialogue context yields lower answer NLL "
        "than an unrelated context, not whether the response is semantically correct. "
        "Random or length-matched contexts from other conversations may accidentally "
        "be plausible, and lexical/distribution cues may contribute. "
        "Comparison uses per-example mean answer-token NLL rather than corpus-token "
        "weighted NLL. Held-out split is per dialogue file, not independent by speaker. "
        "The current 64-token dataset is selected and may not represent longer exchanges."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Report:", args.output)
    return report


def main():
    p = argparse.ArgumentParser(description="v0.8.8 counterfactual context-swap NLL evaluation")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--checkpoint", type=Path, default=Path("model/model-rag-paired-context-v0.8.7.pt"))
    p.add_argument("--test-data", type=Path, default=Path("data/dialogue/real_persona_context_v087/paired_context_test_v087.jsonl"))
    p.add_argument("--sample", type=int, default=200)
    p.add_argument("--length-tolerance", type=int, default=6)
    p.add_argument("--seed", type=int, default=987)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output", type=Path, default=Path("results/context_swap_v088.json"))
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--rope", action="store_true", help="Load RoPE checkpoint explicitly")
    p.add_argument("--overwrite", action="store_true")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
