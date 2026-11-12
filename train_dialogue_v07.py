"""LLM_RAG v0.7 — supervised fine-tuning of short, quotation-derived answers.

Compatible with LLM_GPU v0.3 character tokenizer, context_length=64.
Only answer tokens contribute to CE loss. No audit or approval workflow.
This is synthetic quotation SFT, NOT an independent proof of persona quality.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict) or not all(item.get(k) for k in ("id", "topic", "answer")):
                raise ValueError(f"{path}:{i}: require id, topic, answer")
            rows.append(item)
    return rows


def short_answer(row: dict) -> str:
    """Select a sentence present verbatim in the original quotation."""
    text = str(row["answer"])
    if "「" in text and "」" in text:
        text = text.split("「", 1)[1].rsplit("」", 1)[0]
    sentences = [x.strip() for x in text.split("。") if x.strip()]
    topic = str(row["topic"])
    chosen = next((s for s in sentences if topic in s), sentences[0] if sentences else "")
    return chosen + "。" if chosen else ""


def prompt_for(topic: str) -> str:
    return "問:" + topic + "\n答:"


def make_supervised_sequence(tokenizer, topic: str, answer: str,
                             context: int, min_answer_tokens: int = 8):
    """(input ids, target ids, answer-target mask).

    The question and answer prefix are completely present; any overflowing
    answer is shortened, rather than truncating the question. Targets for the
    prefix receive ignore_index=-100. EOS is included if space permits.
    """
    prefix = tokenizer.encode(prompt_for(topic), add_bos=True)
    body = tokenizer.encode(answer)
    if len(prefix) >= context - min_answer_tokens:
        return None
    if not body:
        return None
    max_body = context - len(prefix)
    body = body[:max_body]
    sequence = prefix + body
    if len(sequence) < context + 1:
        sequence.append(tokenizer.eos_id)
    sequence = sequence[:context + 1]
    x, y = sequence[:-1], sequence[1:]
    # target index len(prefix)-1 predicts the FIRST answer token.
    labels = [tok if i >= len(prefix) - 1 else -100 for i, tok in enumerate(y)]
    if sum(item != -100 for item in labels) < min_answer_tokens:
        return None
    return x, labels


def prepare_rows(rows, tokenizer, context: int, max_unk_rate: float):
    accepted, skipped, unknown, total = [], 0, 0, 0
    for item in rows:
        topic, answer = str(item["topic"]), short_answer(item)
        pair = make_supervised_sequence(tokenizer, topic, answer, context)
        if pair is None:
            skipped += 1
            continue
        x, y = pair
        unk = x.count(tokenizer.unk_id)
        if unk / len(x) > max_unk_rate:
            skipped += 1
            continue
        unknown += unk
        total += len(x)
        accepted.append((x, y, item["id"]))
    return accepted, skipped, unknown, total


def train(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs and batch-size must be positive; learning-rate > 0")
    if args.checkpoint.resolve() == args.output.resolve():
        raise ValueError("Do not replace the input checkpoint")
    if args.train_data.resolve() == args.val_data.resolve():
        raise ValueError("Train and validation files must be distinct")
    if args.output.resolve() == args.report.resolve():
        raise ValueError("Checkpoint and report must be distinct")
    for output in (args.output, args.report):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} already exists; use --overwrite explicitly")

    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from model import LanguageModel
    from tokenizer import Tokenizer
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    model, source = LanguageModel.load_checkpoint(str(args.checkpoint), device=device)
    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError("Vocab size mismatch")
    train_rows = read_jsonl(args.train_data)
    val_rows = read_jsonl(args.val_data)
    ids_train = {r["id"] for r in train_rows}
    ids_val = {r["id"] for r in val_rows}
    if len(ids_train) != len(train_rows) or len(ids_val) != len(val_rows) or ids_train & ids_val:
        raise ValueError("Duplicate or overlapping training/validation IDs")
    if not train_rows or not val_rows:
        raise ValueError("Empty dataset")
    train_pairs, skip_train, unka, tota = prepare_rows(train_rows, tokenizer, model.context_length, args.max_unk_rate)
    val_pairs, skip_val, unkb, totb = prepare_rows(val_rows, tokenizer, model.context_length, args.max_unk_rate)
    if not train_pairs or not val_pairs:
        raise ValueError("No usable short-answer sequences. Check context and tokenizer.")
    print(f"Device={device} context={model.context_length} vocab={model.vocab_size}")
    print(f"Rows train/validation={len(train_pairs)}/{len(val_pairs)} skipped={skip_train}/{skip_val}")
    print(f"UNK rate accepted={(unka+unkb)/max(1,tota+totb):.2%}")

    def collate(batch):
        xs, ys = [], []
        width = max(len(x) for x, y, _ in batch)
        for x, y, _ in batch:
            xs.append(x + [tokenizer.pad_id] * (width-len(x)))
            ys.append(y + [-100] * (width-len(y)))
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    tr = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    va = DataLoader(val_pairs, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    def evaluate():
        model.eval()
        weighted_loss, n = 0.0, 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                           ignore_index=-100, reduction="sum")
                count = int((y != -100).sum().item())
                weighted_loss += float(loss_sum.item())
                n += count
        return weighted_loss / max(n, 1)

    baseline = evaluate()
    best = baseline
    history = []
    saved = False
    print(f"Baseline held-out-row answer NLL={baseline:.5f}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss, batches = 0.0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            training_loss += float(loss.item())
            batches += 1
        validation = evaluate()
        history.append({"epoch": epoch, "train_answer_nll": training_loss / batches, "val_answer_nll": validation})
        print(f"Epoch {epoch}/{args.epochs}: train_answer_nll={training_loss/batches:.5f} val_answer_nll={validation:.5f}")
        if validation < best:
            best = validation
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"format": "llm-rag-synthetic-sft-v0.7",
                        "config": model.config(), "model_state_dict": model.state_dict(),
                        "loss": validation, "epoch": epoch}, str(args.output))
            saved = True

    report = {
        "checkpoint": str(args.checkpoint), "output": str(args.output),
        "train_examples": len(train_pairs), "validation_examples": len(val_pairs),
        "skipped_train": skip_train, "skipped_val": skip_val,
        "baseline_answer_nll": baseline, "best_answer_nll": best,
        "saved": saved, "epochs": history,
        "scope": "Synthetic quotation-derived short-answer SFT; validation shares source corpus and is NOT an independent holdout.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved best checkpoint:" if saved else "No validation gain; checkpoint not written:", args.output)
    print("Report:", args.report)
    return report


def main():
    p = argparse.ArgumentParser(description="Fine-tune LLM_GPU on synthetic short-answer dialogue")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--checkpoint", type=Path, default=Path("model/model-rag-nagato-v0.2.pt"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--train-data", type=Path, default=Path("data/dialogue/nagato_synthetic_train_v06.jsonl"))
    p.add_argument("--val-data", type=Path, default=Path("data/dialogue/nagato_synthetic_val_v06.jsonl"))
    p.add_argument("--output", type=Path, default=Path("model/model-rag-dialogue-v0.7.pt"))
    p.add_argument("--report", type=Path, default=Path("results/dialogue_sft_v07.json"))
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=0.00005)
    p.add_argument("--max-unk-rate", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
