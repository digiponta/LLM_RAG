"""v0.9.7 controlled ablation: answer SFT vs answer SFT + weak intent loss.

Both arms start from EXACTLY the same existing RoPE Medium checkpoint and
same freshly initialized intent head; pass --aux-weight 0 or 0.2. The intent
heuristics are weak labels, not human-reviewed semantic truth. The auxiliary
head is not used as an answer oracle.

No raw dialogue is written into the JSON report or sent over the network.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from intent_model_v097 import INTENTS, UNKNOWN_INTENT, initialize_intent_model, weak_intent
from train_real_dialogue_v082 import (
    load_dialogue_rows, encode_dialogue, collate_dialogue,
    evaluate_answer_nll,
)


def prepare_labeled(rows, tokenizer, context_length, max_unk_fraction=.10):
    pairs, intent_labels, counts = [], [], Counter()
    for row in rows:
        pair, status = encode_dialogue(row, tokenizer, context_length, max_unk_fraction)
        counts[status] += 1
        if pair is not None:
            pairs.append(pair)
            label = weak_intent(row["question"])
            intent_labels.append(label)
            counts["weak_label_accepted" if label != UNKNOWN_INTENT else "weak_label_masked"] += 1
            if label != UNKNOWN_INTENT:
                counts[f"class_{INTENTS[label]}"] += 1
    return list(zip(pairs, intent_labels)), dict(counts)


def collate_intent(items, pad_id):
    import torch
    pairs = [pair for pair, _ in items]
    ids, labels = collate_dialogue(pairs, pad_id)
    positions = (labels != -100).to(torch.int64).argmax(dim=1)
    intents = torch.tensor([label for _, label in items], dtype=torch.long)
    return ids, labels, positions, intents


def training_losses(model, ids, labels, positions, intents, aux_weight):
    import torch
    import torch.nn.functional as F
    lm_logits, intent_logits = model.forward_with_intent(ids, positions)
    answer_loss = F.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]),
                                  labels.reshape(-1), ignore_index=-100)
    valid = intents != UNKNOWN_INTENT
    # If no weak labels in a batch, intent loss exactly 0 (no NaN).
    intent_loss = (F.cross_entropy(intent_logits[valid], intents[valid])
                   if bool(valid.any()) else answer_loss.new_zeros(()))
    return answer_loss + aux_weight * intent_loss, answer_loss, intent_loss


def validate(model, loader, device):
    import torch
    import torch.nn.functional as F
    loss_total = 0.0
    answer_tokens = 0
    correct = 0
    labeled = 0
    model.eval()
    with torch.no_grad():
        for ids, labels, positions, intents in loader:
            ids, labels = ids.to(device), labels.to(device)
            positions, intents = positions.to(device), intents.to(device)
            lm_logits, intent_logits = model.forward_with_intent(ids, positions)
            token_sum = F.cross_entropy(
                lm_logits.reshape(-1, lm_logits.shape[-1]),
                labels.reshape(-1), ignore_index=-100, reduction="sum"
            )
            loss_total += float(token_sum)
            answer_tokens += int((labels != -100).sum().item())
            valid = intents != UNKNOWN_INTENT
            labeled += int(valid.sum().item())
            if bool(valid.any()):
                correct += int((intent_logits[valid].argmax(dim=-1) == intents[valid]).sum().item())
    if answer_tokens == 0:
        raise ValueError("No answer tokens for validation")
    return {"answer_nll": loss_total / answer_tokens,
            "weak_intent_accuracy": correct / labeled if labeled else None,
            "weak_intent_labeled": labeled}


def train(args):
    import torch
    from torch.utils.data import DataLoader
    if not (0 <= args.aux_weight <= 10):
        raise ValueError("aux-weight must be in [0,10]")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size and learning-rate must be positive")
    if args.max_train < 0 or args.max_validation < 0:
        raise ValueError("sample limits must be nonnegative")
    paths = [args.checkpoint, args.source, args.validation, args.tokenizer]
    if args.source.resolve() == args.validation.resolve():
        raise ValueError("Train/validation data must differ")
    for target in (args.output, args.report):
        if target.resolve() in {p.resolve() for p in paths}:
            raise ValueError("Output cannot overwrite an input")
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"Existing output: {target}")
    if args.output.resolve() == args.report.resolve():
        raise ValueError("Output and report must differ")

    from tokenizer import Tokenizer
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = Tokenizer.load(str(args.tokenizer))
    model, ckpt = initialize_intent_model(args.checkpoint, device=device, seed=args.seed)
    if model.vocab_size != tok.vocab_size:
        raise ValueError("Tokenizer vocab mismatch")

    tr_rows = load_dialogue_rows(args.source, args.max_train, args.seed)
    va_rows = load_dialogue_rows(args.validation, args.max_validation, args.seed + 1)
    if ({r["id"] for r in tr_rows} & {r["id"] for r in va_rows} or
        {r["conversation_id"] for r in tr_rows} & {r["conversation_id"] for r in va_rows}):
        raise ValueError("Train/validation overlap")
    tr, train_counts = prepare_labeled(tr_rows, tok, model.context_length)
    va, val_counts = prepare_labeled(va_rows, tok, model.context_length)
    if not tr or not va:
        raise ValueError("No valid pairs")
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                           collate_fn=lambda items: collate_intent(items, tok.pad_id))
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                           collate_fn=lambda items: collate_intent(items, tok.pad_id))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    initial = validate(model, va_loader, device)
    best = initial["answer_nll"]
    history = []
    saved = False
    print(f"Device={device} context={model.context_length} params={model.parameter_count} aux_weight={args.aux_weight}")
    print(f"Sampled: train={len(tr_rows)} validation={len(va_rows)}")
    print(f"Encoded: train={len(tr)} {train_counts}; validation={len(va)} {val_counts}")
    print(f"Initial: val_answer_nll={best:.5f} weak_intent_accuracy={initial['weak_intent_accuracy']}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        batches = 0
        for ids, labels, positions, intents in tr_loader:
            ids, labels = ids.to(device), labels.to(device)
            positions, intents = positions.to(device), intents.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, answer, auxiliary = training_losses(model, ids, labels, positions, intents, args.aux_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += float(answer.detach())
            batches += 1
        scores = validate(model, va_loader, device)
        history.append({"epoch": epoch, "train_batch_mean_answer_nll": train_sum / batches, **scores})
        print(f"Epoch {epoch}/{args.epochs}: train={train_sum/batches:.5f} val_nll={scores['answer_nll']:.5f} "
              f"weak_intent_acc={scores['weak_intent_accuracy']}")
        if scores["answer_nll"] < best:
            best = scores["answer_nll"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"format": "llm-rag-intent-aux-v0.9.7",
                        "config": model.config(), "model_state_dict": model.state_dict(),
                        "epoch": epoch, "loss": best, "aux_weight": args.aux_weight},
                       args.output)
            saved = True
    report = {
        "version": "v0.9.7", "checkpoint": str(args.output),
        "source_checkpoint": str(args.checkpoint), "aux_weight": args.aux_weight,
        "seed": args.seed, "training_rows": len(tr), "validation_rows": len(va),
        "training_label_counts": train_counts, "validation_label_counts": val_counts,
        "initial_validation": initial, "best_answer_nll": best, "saved": saved,
        "history": history,
        "limitations": (
            "Weak regex labels are not human-verified intent annotations. "
            "Intent accuracy is measured against the same pseudo-labeling rules. "
            "Generative quality and factual correctness are not assessed. "
            "The auxiliary head is not used to route answers. "
            "Test split and private raw dialogue are not read or included in this report."),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Checkpoint:", args.output if saved else "not saved (no validation gain)")
    print("Report:", args.report)
    return report


def main():
    p = argparse.ArgumentParser(description="Controlled dialogue intent auxiliary-loss ablation")
    p.add_argument("--checkpoint", type=Path, default=Path("model/model-rag-rope-medium-v0.9.5.pt"))
    p.add_argument("--source", type=Path, default=Path("data/dialogue/real_persona_structured_v092/paired_legacy_train_v092.jsonl"))
    p.add_argument("--validation", type=Path, default=Path("data/dialogue/real_persona_structured_v092/paired_legacy_validation_v092.jsonl"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--output", type=Path, default=Path("model/model-rag-intent-aux-v0.9.7.pt"))
    p.add_argument("--report", type=Path, default=Path("results/intent_aux_train_v097.json"))
    p.add_argument("--aux-weight", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.00002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-validation", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--overwrite", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
