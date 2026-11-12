"""Fine-tune existing LLM_GPU checkpoint on the Nagato corpus.

No audit/approval pipeline. Does not overwrite the original weights or tokenizer.
Uses the same tokenization as the checkpoint. This is continued LM training, NOT
supervised dialogue instruction tuning.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def make_windows(ids: list[int], context: int, stride: int) -> list[tuple[list[int], list[int]]]:
    """Do not mix targets across document boundaries."""
    if context < 2 or stride < 1:
        raise ValueError("context must be >=2 and stride >=1")
    if len(ids) <= context:
        return []
    starts = list(range(0, len(ids)-context, stride))
    last = len(ids)-context-1
    if starts[-1] != last:
        starts.append(last)
    return [(ids[i:i+context], ids[i+1:i+context+1]) for i in starts]


def load_nagato_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    # Skip corpus attribution header; preserve each post as an independent unit.
    if lines and lines[0].startswith("@NagatFakeBot Tweet Messages"):
        lines = lines[1:]
    return [line.strip() for line in lines if line.strip()]


def train(args) -> dict:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    gpu_dir = args.gpu_dir.resolve()
    sys.path.insert(0, str(gpu_dir))
    from model import LanguageModel
    from tokenizer import Tokenizer

    if args.epochs < 1 or args.batch_size < 1 or args.stride < 1:
        raise ValueError("epochs, batch-size, stride must be >=1")
    if not 0 < args.val_fraction < 0.5:
        raise ValueError("val-fraction must be between 0 and 0.5")
    if args.output.resolve() == args.checkpoint.resolve():
        raise ValueError("Output must not replace the source checkpoint.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --overwrite explicitly")

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    model, checkpoint = LanguageModel.load_checkpoint(str(args.checkpoint), device=device)
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint/tokenizer vocabulary mismatch")

    lines = load_nagato_lines(args.data)
    if len(lines) < 3:
        raise ValueError("Need at least 3 nonempty corpus lines.")
    rng = random.Random(args.seed)
    rng.shuffle(lines)
    val_size = max(1, round(len(lines) * args.val_fraction))
    val_lines, train_lines = lines[:val_size], lines[val_size:]

    def encode_lines(items):
        results = []
        unknown = total = 0
        for line in items:
            ids = tokenizer.encode(line, add_bos=True, add_eos=True)
            unknown += ids.count(tokenizer.unk_id)
            total += len(ids)
            # Do not train across unrelated posts.
            results.extend(make_windows(ids, model.context_length, args.stride))
        return results, unknown, total

    train_windows, unk_train, n_train = encode_lines(train_lines)
    val_windows, unk_val, n_val = encode_lines(val_lines)
    unk_rate = (unk_train + unk_val) / max(1, n_train + n_val)
    print(f"Device={device} context={model.context_length} vocab={tokenizer.vocab_size}")
    print(f"Posts train/validation: {len(train_lines)}/{len(val_lines)}")
    print(f"Windows train/validation: {len(train_windows)}/{len(val_windows)}")
    print(f"Tokenizer UNK rate: {unk_rate:.2%}")
    if unk_rate > args.max_unk_rate:
        raise ValueError(f"UNK rate {unk_rate:.2%} exceeds {args.max_unk_rate:.2%}. "
                         "Use matching tokenizer, or retrain a new model/tokenizer pair.")
    if not train_windows or not val_windows:
        raise ValueError("Not enough data windows for training/validation with this context.")

    class WindowDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            x, y = self.rows[idx]
            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    train_loader = DataLoader(WindowDataset(train_windows), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(WindowDataset(val_windows), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.to(device).reshape(-1))
                losses.append(float(loss.item()))
        return sum(losses) / len(losses)

    base_val = evaluate()
    print(f"Baseline validation NLL: {base_val:.5f}")
    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        train_loss = sum(losses)/len(losses)
        val_loss = evaluate()
        history.append({"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss})
        print(f"Epoch {epoch}/{args.epochs}: train_nll={train_loss:.5f} val_nll={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Same checkpoint config/model weights; exclude optimizer to keep it small.
            torch.save({"format": "llm-rag-nagato-finetune-v0.2",
                        "config": model.config(),
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch, "loss": val_loss}, args.output)
    report = {
        "source_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(args.output),
        "source_loss": checkpoint.get("loss"),
        "baseline_validation_nll": base_val,
        "best_validation_nll": best_val,
        "train_posts": len(train_lines),
        "validation_posts": len(val_lines),
        "train_windows": len(train_windows),
        "validation_windows": len(val_windows),
        "unknown_rate": unk_rate, "history": history,
        "note": "Continuation language-model training, not question-answer fine tuning.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite report: {args.report}")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint: {args.output}")
    print(f"Saved metrics: {args.report}")
    return report


def main():
    p = argparse.ArgumentParser(description="Fine-tune LLM_GPU on data-nagato.txt")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--checkpoint", type=Path, default=Path("../LLM_GPU/model/model-gpu-v0.3.pt"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--data", type=Path, default=Path("data/knowledge/data-nagato.txt"))
    p.add_argument("--output", type=Path, default=Path("model/model-rag-nagato-v0.2.pt"))
    p.add_argument("--report", type=Path, default=Path("results/nagato_finetune_v02.json"))
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--max-unk-rate", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
