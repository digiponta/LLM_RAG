"""Evaluate v0.7 vs v0.8.2 on local held-out-by-dialogue Real Persona Chat test data.

Scores response-token NLL and exact response reproduction. Writes a local
report, excluding raw private dialogue text by default. This is not a
human-rated semantic quality benchmark.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from train_real_dialogue_v082 import (
    load_dialogue_rows, encode_dialogue, collate_dialogue,
    evaluate_answer_nll,
)
from prepare_real_persona_v081 import compact_prompt


def evaluate(args):
    import torch
    from torch.utils.data import DataLoader

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Existing report: {args.output}")
    if args.sample < 1 or args.generations < 0 or args.max_new_tokens < 1:
        raise ValueError("sample and max-new-tokens must be positive; generations >=0")
    if args.test_data.resolve() in {args.baseline.resolve(), args.tuned.resolve(), args.output.resolve()}:
        raise ValueError("Test data path must be separate")
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from tokenizer import Tokenizer
    from mha_rope_v096 import load_model_v096
    from intent_model_v097 import load_intent_model

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    tokenizer = Tokenizer.load(str(args.tokenizer))
    rows = load_dialogue_rows(args.test_data, args.sample, args.seed)
    models = {}
    for name, checkpoint_path, use_rope in (
        ("baseline", args.baseline, getattr(args, 'baseline_rope', False)),
        ("tuned", args.tuned, getattr(args, 'tuned_rope', False)),
    ):
        is_mha = getattr(args, "baseline_mha", False) if name == "baseline" else getattr(args, "tuned_mha", False)
        if is_mha and not use_rope:
            raise ValueError(f"{name}: --{name}-mha requires --{name}-rope")
        architecture = "mha" if is_mha else ("rope" if use_rope else "legacy")
        use_intent = (getattr(args, "baseline_intent", False)
                      if name == "baseline" else getattr(args, "tuned_intent", False))
        if use_intent:
            if not use_rope or is_mha:
                raise ValueError(f"{name}: intent checkpoint requires single-head RoPE")
            model, _ = load_intent_model(checkpoint_path, device)
        else:
            model, _ = load_model_v096(checkpoint_path, device, expected=architecture,
                                        mha_heads=getattr(args, "mha_heads", 4))
        if model.vocab_size != tokenizer.vocab_size:
            raise ValueError(f"{name}: tokenizer mismatch")
        models[name] = model
    context_limit = min(model.context_length for model in models.values())
    pairs, usable_rows = [], []
    stats = {"over_context": 0, "high_unk": 0, "too_short": 0}
    for row in rows:
        pair, status = encode_dialogue(row, tokenizer, context_limit, args.max_unk_fraction)
        if pair is None:
            stats[status] += 1
        else:
            pairs.append(pair)
            usable_rows.append(row)
    if not pairs:
        raise ValueError("No usable test rows")
    loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda batch: collate_dialogue(batch, tokenizer.pad_id))
    scores = {name: {"answer_token_nll": evaluate_answer_nll(model, loader, device)}
              for name, model in models.items()}
    samples = []
    for row in usable_rows[:args.generations]:
        entry = {"case_id": row["id"]}
        prompt = tokenizer.encode(compact_prompt(row), add_bos=True)
        for name, model in models.items():
            generated = model.generate(prompt,
                                       max_new_tokens=args.max_new_tokens,
                                       eos_id=tokenizer.eos_id, temperature=0,
                                       top_k=None, repetition_penalty=1.3)
            tokens = generated[len(prompt):]
            response = tokenizer.decode(tokens, skip_special_tokens=True)
            entry[name] = {
                "exact_match": response == row["answer"],
                "generated_chars": len(response),
                "stopped_eos": bool(tokens and tokens[-1] == tokenizer.eos_id),
            }
            # Raw text is sensitive: include it only by explicit option.
            if args.include_text:
                entry[name]["response"] = response
        if args.include_text:
            entry["question"] = row["question"]
            entry["reference_answer"] = row["answer"]
        samples.append(entry)
    for name in models:
        scores[name]["exact_match_in_sample"] = sum(case[name]["exact_match"] for case in samples)
        scores[name]["sample_count"] = len(samples)
    report = {
        "version": "v0.8.2", "test_rows_selected": len(rows), "test_rows_usable": len(pairs),
        "skipped": stats, "models": scores, "samples": samples,
        "text_included": bool(args.include_text),
        "limitations": ("Real Persona Chat test partition is held out by dialogue file from converter. "
                        "It may still overlap speakers, topics and near-identical conversations; "
                        "not a human-rated, fully independent evaluation."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"test_rows_selected": len(rows), "test_rows_usable": len(pairs),
                      "skipped": stats, "models": scores, "report": str(args.output)},
                     ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate real dialogue fine-tuning")
    parser.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    parser.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    parser.add_argument("--baseline", type=Path, default=Path("model/model-rag-dialogue-v0.7.pt"))
    parser.add_argument("--tuned", type=Path, default=Path("model/model-rag-real-dialogue-v0.8.2.pt"))
    parser.add_argument("--test-data", type=Path, default=Path("data/dialogue/real_persona_v081/real_persona_test_v081.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/real_dialogue_eval_v082.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-unk-fraction", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--baseline-intent", action="store_true",
                        help="Load baseline as v0.9.7 auxiliary intent checkpoint")
    parser.add_argument("--tuned-intent", action="store_true",
                        help="Load tuned as v0.9.7 auxiliary intent checkpoint")
    parser.add_argument("--baseline-mha", action="store_true",
                        help="Baseline checkpoint uses multi-head RoPE v0.9.6")
    parser.add_argument("--tuned-mha", action="store_true",
                        help="Tuned checkpoint uses multi-head RoPE v0.9.6")
    parser.add_argument("--mha-heads", type=int, default=4)
    parser.add_argument("--baseline-rope", action="store_true",
                        help="Evaluate baseline with RoPE attention")
    parser.add_argument("--tuned-rope", action="store_true",
                        help="Evaluate tuned checkpoint with RoPE attention")
    parser.add_argument("--overwrite", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
