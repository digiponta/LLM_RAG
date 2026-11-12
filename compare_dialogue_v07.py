"""Compare baseline and v0.7 dialogue generation without reference leakage.

This is a generation smoke test, NOT a factual accuracy or independent
generalization benchmark. Prompts are fixed and reported verbatim.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from train_dialogue_v07 import prompt_for

DEFAULT_TOPICS = ["宇宙", "時間", "量子力学", "観測", "粒子", "情報", "論理", "人間", "空間", "読書"]


def has_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text))


def run(args):
    import torch
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from model import LanguageModel
    from tokenizer import Tokenizer
    for path in (args.baseline, args.tuned):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing overwrite: {args.output}")
    tokenizer = Tokenizer.load(str(args.tokenizer))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    models = {}
    for name, path in (("baseline", args.baseline), ("tuned", args.tuned)):
        model, _ = LanguageModel.load_checkpoint(str(path), device=device)
        if model.vocab_size != tokenizer.vocab_size:
            raise ValueError(f"Vocabulary mismatch for {name}")
        models[name] = model
    topics = args.topics or DEFAULT_TOPICS
    rows = []
    for topic in topics:
        prompt = prompt_for(topic)
        ids = tokenizer.encode(prompt, add_bos=True)
        if len(ids) >= min(m.context_length for m in models.values()):
            raise ValueError(f"Prompt too long for model context: {topic}")
        item = {"topic": topic, "prompt": prompt}
        for name, model in models.items():
            output = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                    eos_id=tokenizer.eos_id, temperature=0,
                                    top_k=None, repetition_penalty=1.0)
            answer = tokenizer.decode(output[len(ids):], skip_special_tokens=True)
            item[name] = {
                "text": answer, "nonempty": bool(answer.strip()),
                "contains_topic": topic in answer,
                "contains_japanese": has_japanese(answer),
                "length_chars": len(answer),
            }
        rows.append(item)
        print(f"[{topic}] baseline={item['baseline']['text']!r}")
        print(f"         tuned={item['tuned']['text']!r}")
    summary = {}
    for name in models:
        summary[name] = {
            "nonempty": sum(r[name]["nonempty"] for r in rows),
            "contains_japanese": sum(r[name]["contains_japanese"] for r in rows),
            "contains_topic": sum(r[name]["contains_topic"] for r in rows),
        }
    report = {
        "device": str(device), "baseline": str(args.baseline), "tuned": str(args.tuned),
        "results": rows, "summary_counts": summary,
        "warning": "Diagnostic text metrics only. These prompts and source corpus are not a truly independent quality benchmark. Evaluate coherence and correctness manually.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Report:", args.output)
    return report


def main():
    p = argparse.ArgumentParser(description="Compare short-answer generations")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--baseline", type=Path, default=Path("model/model-rag-nagato-v0.2.pt"))
    p.add_argument("--tuned", type=Path, default=Path("model/model-rag-dialogue-v0.7.pt"))
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("results/dialogue_compare_v07.json"))
    p.add_argument("--topics", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.max_new_tokens < 1:
        p.error("max-new-tokens must be >= 1")
    run(args)


if __name__ == "__main__":
    main()
