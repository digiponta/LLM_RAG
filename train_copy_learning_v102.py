"""LLM_RAG v0.10.2 — frozen-baseline -> small copy-task overfit probe.

Unlike the general SFT trainer, saves a FINAL checkpoint unconditionally:
micro-overfit can lower train NLL while worsening development NLL. A validation
gated save would conceal the very case we seek to diagnose.

Uses exact v0.8.2 answer-only masked loss and same single-head RoPE architecture.
Original checkpoint is never modified. Nagato test sources are never loaded.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from prepare_real_persona_v081 import compact_prompt
from train_real_dialogue_v082 import (
    load_dialogue_rows, prepare_sequences, collate_dialogue, evaluate_answer_nll)
from evaluate_evidence_copy_v101 import score_copy, first_token_diagnostic


def validate_partitions(train, dev):
    if not train or not dev:
        raise ValueError("Nonempty train and development data required")
    if {r["id"] for r in train} & {r["id"] for r in dev}:
        raise ValueError("Duplicate row IDs")
    if {r["conversation_id"] for r in train} & {r["conversation_id"] for r in dev}:
        raise ValueError("Original source lines leaked across partitions")
    for r in [*train, *dev]:
        if r.get("example_type") != "evidence_copy":
            raise ValueError("Only explicit evidence-copy rows supported")
        if r.get("context") != "資料:" + r["answer"]:
            raise ValueError("Copy targets must equal supplied evidence")


def measure_copy(model, tok, rows, *, max_new_tokens=40):
    import torch
    counts = {"cases": len(rows), "exact": 0, "first_token": 0,
              "mean_prefix_coverage": 0.0, "mean_position_accuracy": 0.0,
              "mean_generated_chars": 0.0}
    model.eval()
    for r in rows:
        prefix = tok.encode(compact_prompt(r), add_bos=True)
        expected = tok.encode(r["answer"])
        first = first_token_diagnostic(model, prefix, expected)
        with torch.inference_mode():
            generated = model.generate(
                prefix, max_new_tokens=max(max_new_tokens, len(expected) + 8),
                eos_id=tok.eos_id, temperature=0, top_k=None, repetition_penalty=1.0)
        tail = generated[len(prefix):]
        actual = tok.decode(tail, skip_special_tokens=True)
        metrics = score_copy(r["answer"], actual,
                             eos=bool(tail and tail[-1] == tok.eos_id))
        counts["exact"] += int(metrics["exact_match"])
        counts["first_token"] += int(first["correct"])
        counts["mean_prefix_coverage"] += metrics["prefix_coverage"]
        counts["mean_position_accuracy"] += metrics["position_accuracy"]
        counts["mean_generated_chars"] += metrics["generated_chars"]
    for key in ("mean_prefix_coverage", "mean_position_accuracy", "mean_generated_chars"):
        counts[key] /= counts["cases"]
    counts["exact_rate"] = counts["exact"] / counts["cases"]
    counts["first_token_accuracy"] = counts["first_token"] / counts["cases"]
    return counts


def run(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tokenizer import Tokenizer
    from mha_rope_v096 import load_model_v096

    if (args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0
            or args.eval_every < 1 or args.max_new_tokens < 1):
        raise ValueError("Invalid training options")
    sources = [args.train, args.development, args.checkpoint, args.tokenizer]
    destinations = [args.output, args.report]
    if len({x.resolve() for x in sources + destinations}) != len(sources) + len(destinations):
        raise ValueError("Input/output path collision")
    if not args.overwrite and any(x.exists() for x in destinations):
        raise FileExistsError("One or more outputs exist")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tok = Tokenizer.load(str(args.tokenizer))
    model, _ = load_model_v096(args.checkpoint, device, expected="rope")
    if tok.vocab_size != model.vocab_size:
        raise ValueError("Tokenizer and model mismatch")
    train_rows = load_dialogue_rows(args.train, 0, args.seed)
    dev_rows = load_dialogue_rows(args.development, 0, args.seed)
    validate_partitions(train_rows, dev_rows)
    train_pairs, train_stats = prepare_sequences(train_rows, tok, model.context_length, .1)
    dev_pairs, dev_stats = prepare_sequences(dev_rows, tok, model.context_length, .1)
    if len(train_pairs) != len(train_rows) or len(dev_pairs) != len(dev_rows):
        raise ValueError(f"Copy rows filtered unexpectedly: train={train_stats}, dev={dev_stats}")
    loader = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda batch: collate_dialogue(batch, tok.pad_id))
    train_eval = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda batch: collate_dialogue(batch, tok.pad_id))
    dev_eval = DataLoader(dev_pairs, batch_size=args.batch_size, shuffle=False,
                          collate_fn=lambda batch: collate_dialogue(batch, tok.pad_id))
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    before = {
        "train_nll": evaluate_answer_nll(model, train_eval, device),
        "development_nll": evaluate_answer_nll(model, dev_eval, device),
        "train_generation": measure_copy(model, tok, train_rows,
                                         max_new_tokens=args.max_new_tokens),
        "development_generation": measure_copy(model, tok, dev_rows,
                                               max_new_tokens=args.max_new_tokens),
    }
    print(json.dumps({"phase": "before", **before}, ensure_ascii=False))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            row = {
                "epoch": epoch,
                "training_batch_loss": sum(losses) / len(losses),
                "train_nll": evaluate_answer_nll(model, train_eval, device),
                "development_nll": evaluate_answer_nll(model, dev_eval, device),
                "train_generation": measure_copy(model, tok, train_rows,
                                                 max_new_tokens=args.max_new_tokens),
                "development_generation": measure_copy(model, tok, dev_rows,
                                                       max_new_tokens=args.max_new_tokens),
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Intentional: this is a diagnostic FINAL snapshot, even if dev NLL got worse.
    torch.save({
        "format": "llm-rag-copy-learning-probe-v0.10.2",
        "config": model.config(), "model_state_dict": model.state_dict(),
        "loss": history[-1]["development_nll"], "epoch": args.epochs,
    }, args.output)
    report = {
        "version": "v0.10.2", "initial_checkpoint": str(args.checkpoint),
        "final_checkpoint": str(args.output), "train_size": len(train_rows),
        "development_size": len(dev_rows), "before": before,
        "history": history,
        "qualification": (
            "Training and development examples use distinct original Nagato source lines. "
            "No final Nagato test examples are loaded. Saved checkpoint is the final "
            "epoch regardless of validation regression; this is a diagnostic overfit "
            "probe, not a recommended general dialogue model. All sample text is kept "
            "in local input JSONL only, not this report."),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "completed", "train_generation": history[-1]["train_generation"],
                      "development_generation": history[-1]["development_generation"],
                      "checkpoint": str(args.output), "report": str(args.report)},
                     ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="10-char copy-learning micro overfit experiment")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--checkpoint", type=Path, default=Path(
        "model/model-rag-binding-v0.10.0.pt"))
    p.add_argument("--train", type=Path, default=Path(
        "data/dialogue/copy_probe_v102/copy_train_v102.jsonl"))
    p.add_argument("--development", type=Path, default=Path(
        "data/dialogue/copy_probe_v102/copy_development_v102.jsonl"))
    p.add_argument("--output", type=Path, default=Path(
        "model/model-rag-copy-probe-v0.10.2.pt"))
    p.add_argument("--report", type=Path, default=Path(
        "results/copy_learning_probe_v102.json"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=.0002)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    run(args)


if __name__ == "__main__":
    main()
