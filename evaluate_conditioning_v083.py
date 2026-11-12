"""LLM_RAG v0.8.3 question-conditioning diagnostic.

Compare NLL(answer | matching question) versus NLL(same answer |
unrelated question). Positive margin means a conditional preference,
not factual correctness or good dialogue generation. Reads TEST split only.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
from train_real_dialogue_v082 import load_dialogue_rows, encode_dialogue


def swapped_question_rows(rows: list[dict], seed: int = 42) -> list[dict]:
    """Deterministically pair each answer with a different question.

    A cyclic rotation guarantees a different row, though the texts
    themselves may coincide. Reject exact same-question pairs.
    """
    if len(rows) < 2:
        raise ValueError("Need two or more rows")
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    result = []
    for offset, index in enumerate(indices):
        row = dict(rows[index])
        other = rows[indices[(offset + 1) % len(indices)]]
        if row["question"] == other["question"]:
            continue
        row["question"] = other["question"]
        row["context"] = ""  # swapped context would otherwise leak original condition
        result.append((index, row))
    return result


def answer_nll(model, sequence, device) -> float:
    import torch
    import torch.nn.functional as F
    x, y = sequence
    inputs = torch.tensor([x], device=device, dtype=torch.long)
    labels = torch.tensor(y, device=device, dtype=torch.long)
    with torch.no_grad():
        logits = model(inputs)[0]
        nll = F.cross_entropy(logits, labels, ignore_index=-100, reduction="mean")
    return float(nll.item())


def compare_conditions(model, rows, tokenizer, device,
                       seed: int = 42, max_unk_fraction: float = .1) -> dict:
    model.eval()
    margins = []
    valid = skipped = 0
    for index, mismatch in swapped_question_rows(rows, seed):
        matching = rows[index]
        positive, _ = encode_dialogue(matching, tokenizer, model.context_length, max_unk_fraction)
        negative, _ = encode_dialogue(mismatch, tokenizer, model.context_length, max_unk_fraction)
        if positive is None or negative is None:
            skipped += 1
            continue
        good = answer_nll(model, positive, device)
        bad = answer_nll(model, negative, device)
        margins.append(bad - good)
        valid += 1
    if not margins:
        raise ValueError("No matched/mismatched pairs fit the model")
    return {
        "usable_pairs": valid, "skipped": skipped,
        "mean_nll_margin": sum(margins) / len(margins),
        "positive_margin_rate": sum(x > 0 for x in margins) / len(margins),
        "mean_absolute_margin": sum(abs(x) for x in margins) / len(margins),
        "meaning": "margin = NLL(answer | shuffled question) - NLL(answer | matching question)",
    }


def main():
    p = argparse.ArgumentParser(description="Diagnose question-conditioning rather than generic answer frequency")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--baseline", type=Path, default=Path("model/model-rag-real-dialogue-v0.8.2.pt"))
    p.add_argument("--candidate", type=Path, default=None)
    p.add_argument("--test-data", type=Path, default=Path("data/dialogue/real_persona_v081/real_persona_test_v081.jsonl"))
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--max-unk-fraction", type=float, default=.1)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output", type=Path, default=Path("results/question_conditioning_v083.json"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.sample < 2 or not 0 <= args.max_unk_fraction <= 1:
        p.error("sample must be >=2; max-unk-fraction must be in [0,1]")
    if args.output.exists() and not args.overwrite:
        p.error(f"Existing report: {args.output}")
    if args.test_data.resolve() == args.output.resolve():
        p.error("Report must differ from test data")
    import torch
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from model import LanguageModel
    from tokenizer import Tokenizer
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    rows = load_dialogue_rows(args.test_data, args.sample, args.seed)
    checkpoints = {"baseline": args.baseline}
    if args.candidate is not None:
        checkpoints["candidate"] = args.candidate
    results = {}
    for name, checkpoint in checkpoints.items():
        model, _ = LanguageModel.load_checkpoint(str(checkpoint), device=device)
        if model.vocab_size != tokenizer.vocab_size:
            raise ValueError(f"{name}: vocabulary mismatch")
        results[name] = compare_conditions(model, rows, tokenizer, device,
                                           args.seed, args.max_unk_fraction)
    report = {"version": "v0.8.3", "test_rows": len(rows), "results": results,
              "limitations": ("Shuffled-question NLL margin is a proxy for response conditioning, "
                              "not coherence or correctness. Conversation-file-held-out tests "
                              "may overlap by speakers, topics or near duplicate. "
                              "No raw dialogue text included.")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
