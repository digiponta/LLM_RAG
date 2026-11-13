"""LLM_RAG v0.8.2: SFT on locally prepared Real Persona Chat short dialogue pairs.

Use the LLM_GPU v0.3 character tokenizer and causal LanguageModel without
changing model architecture. The loss is computed on answer tokens only.
Source JSONL and source checkpoint are read-only. No external network I/O.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from prepare_real_persona_v081 import compact_prompt


def load_dialogue_rows(path: Path, limit: int, seed: int) -> list[dict]:
    """Load a deterministic random subset, without changing original JSONL."""
    if limit < 0:
        raise ValueError("limit must be >= 0")
    records = []
    ids = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k] for k in ("id", "question", "answer", "conversation_id")):
            raise ValueError(f"{path}:{line_no}: invalid dialogue entry")
        if row["id"] in ids:
            raise ValueError(f"{path}:{line_no}: duplicate id")
        ids.add(row["id"])
        records.append(row)
    if not records:
        raise ValueError(f"No rows in {path}")
    if limit and len(records) > limit:
        records = random.Random(seed).sample(records, limit)
    return records


def encode_dialogue(row, tokenizer, context_length: int,
                    max_unknown_fraction: float = .10, min_answer_tokens: int = 2):
    """Build (x, label) with complete prompt and full response; no truncation.

    For target at x[len(prefix)-1], predict answer's first token. Prefix
    targets are -100. EOS is predicted after the complete answer.
    """
    prefix = tokenizer.encode(compact_prompt(row), add_bos=True)
    answer = tokenizer.encode(row["answer"])
    if not answer or len(answer) < min_answer_tokens:
        return None, "too_short"
    if len(prefix) + len(answer) > context_length:
        return None, "over_context"
    seq = prefix + answer + [tokenizer.eos_id]
    x, targets = seq[:-1], seq[1:]
    if len(x) > context_length:
        return None, "over_context"
    unknowns = x.count(tokenizer.unk_id)
    if unknowns / max(1, len(x)) > max_unknown_fraction:
        return None, "high_unk"
    labels = [-100] * (len(prefix) - 1) + targets[len(prefix)-1:]
    if len(labels) != len(x) or labels[-1] != tokenizer.eos_id:
        raise AssertionError("Malformed target mask")
    return (x, labels), "accepted"


def prepare_sequences(rows, tokenizer, context_length: int, max_unknown_fraction: float):
    pairs, stats = [], Counter()
    for row in rows:
        sequence, reason = encode_dialogue(row, tokenizer, context_length, max_unknown_fraction)
        stats[reason] += 1
        if sequence is not None:
            pairs.append(sequence)
    return pairs, dict(stats)


def collate_dialogue(batch, pad_id):
    import torch
    width = max(len(x) for x, _ in batch)
    xs = [x + [pad_id] * (width-len(x)) for x, y in batch]
    ys = [y + [-100] * (width-len(y)) for x, y in batch]
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def evaluate_answer_nll(model, loader, device):
    import torch
    import torch.nn.functional as F
    model.eval()
    loss_sum, tokens = 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), reduction="sum", ignore_index=-100)
            loss_sum += float(loss.item())
            tokens += int((labels != -100).sum().item())
    if tokens == 0:
        raise ValueError("No evaluation answer tokens")
    return loss_sum / tokens


def train(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size and learning-rate must be positive")
    if args.max_train < 0 or args.max_validation < 0:
        raise ValueError("sample limits must be >=0")
    if not 0 <= args.max_unk_fraction <= 1:
        raise ValueError("max-unk-fraction must be within [0,1]")
    if args.source.resolve() == args.validation.resolve():
        raise ValueError("Training and validation must be separate files")
    guarded = [args.source, args.validation, args.checkpoint, args.tokenizer]
    dests = [args.output, args.report]
    if args.output.resolve() == args.report.resolve() or any(a.resolve() == b.resolve() for a in guarded for b in dests):
        raise ValueError("Output paths must not overwrite inputs")
    for file in dests:
        if file.exists() and not args.overwrite:
            raise FileExistsError(f"Existing output: {file}; use --overwrite")

    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from tokenizer import Tokenizer
    from rope_model_v093 import load_configured_model
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    model, original_ckpt = load_configured_model(
        args.checkpoint, device, rope=args.rope,
        allow_plain_to_rope=args.allow_rope_conversion)
    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError("Tokenizer/checkpoint vocab mismatch")

    train_rows = load_dialogue_rows(args.source, args.max_train, args.seed)
    val_rows = load_dialogue_rows(args.validation, args.max_validation, args.seed + 1)
    if {r["id"] for r in train_rows}.intersection(r["id"] for r in val_rows):
        raise ValueError("Train/validation ID overlap")
    if {r["conversation_id"] for r in train_rows}.intersection(r["conversation_id"] for r in val_rows):
        raise ValueError("Train/validation dialogue overlap")
    train_pairs, stats_train = prepare_sequences(train_rows, tokenizer, model.context_length, args.max_unk_fraction)
    val_pairs, stats_val = prepare_sequences(val_rows, tokenizer, model.context_length, args.max_unk_fraction)
    if not train_pairs or not val_pairs:
        raise ValueError("No valid training/validation sequences; inspect UNK and context stats")
    print(f"Device={device} context={model.context_length} vocab={model.vocab_size}")
    print(f"Sampled: train={len(train_rows)} validation={len(val_rows)}")
    print(f"Encoded: train={len(train_pairs)} {stats_train}; validation={len(val_pairs)} {stats_val}")
    tr = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    collate_fn=lambda batch: collate_dialogue(batch, tokenizer.pad_id))
    va = DataLoader(val_pairs, batch_size=args.batch_size, shuffle=False, num_workers=0,
                    collate_fn=lambda batch: collate_dialogue(batch, tokenizer.pad_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    base_nll = evaluate_answer_nll(model, va, device)
    best_nll = base_nll
    saved = False
    history = []
    print(f"Baseline validation answer NLL: {base_nll:.5f}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, batches = 0.0, 0
        for inputs, labels in tr:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
        val_nll = evaluate_answer_nll(model, va, device)
        history.append({"epoch": epoch, "train_answer_nll": train_loss / batches, "val_answer_nll": val_nll})
        print(f"Epoch {epoch}/{args.epochs}: train={train_loss/batches:.5f} validation={val_nll:.5f}")
        if val_nll < best_nll:
            best_nll = val_nll
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format": "llm-rag-real-dialogue-sft-v0.8.2",
                "config": model.config(), "model_state_dict": model.state_dict(),
                "loss": val_nll, "epoch": epoch,
            }, args.output)
            saved = True
    report = {
        "version": "v0.8.2", "source_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(args.output), "device": str(device),
        "source_train": str(args.source), "source_validation": str(args.validation),
        "selected_train": len(train_rows), "selected_validation": len(val_rows),
        "usable_train": len(train_pairs), "usable_validation": len(val_pairs),
        "filter_train": stats_train, "filter_validation": stats_val,
        "baseline_validation_answer_nll": base_nll, "best_validation_answer_nll": best_nll,
        "saved": saved, "history": history,
        "note": ("This is short-turn SFT on real conversations; AA/AB do not denote Nagato. "
                 "Validation is dialogue-file-separated but may share people/topics/near-duplicates. "
                 "Evaluation test data are not read by training. Loss improvement is not dialogue quality."),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Best checkpoint:" if saved else "No gain; checkpoint not written:", args.output)
    print("Report:", args.report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Real Persona Chat short-response SFT")
    parser.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    parser.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("model/model-rag-dialogue-v0.7.pt"))
    parser.add_argument("--source", type=Path, default=Path("data/dialogue/real_persona_v081/real_persona_train_v081.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/dialogue/real_persona_v081/real_persona_validation_v081.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("model/model-rag-real-dialogue-v0.8.2.pt"))
    parser.add_argument("--report", type=Path, default=Path("results/real_dialogue_sft_v082.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train", type=int, default=10000, help="0 means all")
    parser.add_argument("--max-validation", type=int, default=1000, help="0 means all")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-unk-fraction", type=float, default=.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rope", action="store_true",
                        help="Use RoPE attention instead of legacy attention")
    parser.add_argument("--allow-rope-conversion", action="store_true",
                        help="Explicitly initialize RoPE attention from plain model weights")
    parser.add_argument("--overwrite", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
