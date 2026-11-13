"""v0.10.0: evidence-conditioned question-answer and followup learning.

Builds local JSONL compatible with the existing Real Persona Chat SFT trainer.
Oracle evidence is provided in the input. It is NOT a retrieval benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from prepare_unified_sft_v099 import TOPICS, mix_training_rows, write_jsonl
from train_real_dialogue_v082 import load_dialogue_rows, encode_dialogue

HEAD = "長門コーパスには「"
TAIL = "」と記されています。"


def excerpt_from_v099(row):
    if row.get("source_kind") != "nagato_corpus" or row.get("example_type") != "source_attributed_qa":
        raise ValueError("Expected v0.9.9 attributed Nagato source")
    answer = row.get("answer", "")
    if not answer.startswith(HEAD) or not answer.endswith(TAIL):
        raise ValueError("Invalid source-quote envelope")
    quote = answer[len(HEAD):-len(TAIL)]
    if not quote:
        raise ValueError("Empty quotation")
    return quote


def bound_variants(row, excerpt_chars=56):
    if excerpt_chars < 16:
        raise ValueError("excerpt_chars must be >= 16")
    excerpt = excerpt_from_v099(row)[:excerpt_chars].strip()
    topic = next((x for x in TOPICS if x in excerpt), None)
    if not topic:
        return []
    common = {
        "conversation_id": row["conversation_id"],
        "source_kind": "nagato_corpus",
        "original_line_hash": row.get("original_line_hash", ""),
        "source_line": row.get("source_line"),
        "evidence_origin": row["id"],
        "review_state": "SOURCE_QUOTE_UNVERIFIED",
        "topic_hint": topic,
        "answer": "資料には「" + excerpt + "」と記されています。",
    }
    question = f"長門コーパスでは「{topic}」について何と述べていますか？"
    direct = {**common, "id": "bind-v100:" + row["id"] + ":direct",
              "question": question, "context": "資料:" + excerpt,
              "example_type": "evidence_direct"}
    followup = {**common, "id": "bind-v100:" + row["id"] + ":followup",
                "question": "その記述をもう少し詳しく示してください",
                "context": f"話題:{topic}／資料:{excerpt}／直前:{question}",
                "example_type": "evidence_followup"}
    return [direct, followup]


def prepare_bound(rows, tokenizer, context_length=256, excerpt_chars=56, max_unk_fraction=.1):
    accepted, stats = [], Counter()
    for row in rows:
        if row.get("example_type") != "source_attributed_qa":
            continue
        candidates = bound_variants(row, excerpt_chars)
        if not candidates:
            stats["no_topic"] += 1
        for candidate in candidates:
            pair, status = encode_dialogue(candidate, tokenizer, context_length, max_unk_fraction)
            stats[status] += 1
            if pair is not None:
                accepted.append(candidate)
    return accepted, dict(stats)


def no_evidence(rows, tokenizer, context_length=256, max_unk_fraction=.1):
    output = []
    for row in rows:
        copied = dict(row)
        copied["id"] += ":no-evidence"
        if row["example_type"] == "evidence_direct":
            copied["context"] = ""
        else:
            topic = row["topic_hint"]
            copied["context"] = (
                f"話題:{topic}／直前:長門コーパスでは「{topic}」について何と述べていますか？")
        copied["example_type"] = row["example_type"] + "_no_evidence"
        pair, _ = encode_dialogue(copied, tokenizer, context_length, max_unk_fraction)
        if pair is None:
            raise ValueError("No-evidence ablation unexpectedly unencodable")
        output.append(copied)
    return output


def check_partitions(train, validation, test):
    ids, origins = set(), {}
    for split, rows in (("train", train), ("validation", validation), ("test", test)):
        for row in rows:
            if row["id"] in ids:
                raise ValueError("Duplicate id")
            ids.add(row["id"])
            origin = row["conversation_id"]
            if origin in origins and origins[origin] != split:
                raise ValueError("Cross-split source leakage")
            origins[origin] = split


def generate(args):
    from tokenizer import Tokenizer
    tok = Tokenizer.load(str(args.tokenizer))
    if not 0 <= args.max_unk_fraction <= 1 or args.context_length < 64:
        raise ValueError("Invalid tokenizer or context options")
    inputs = [args.persona_train, args.persona_validation, args.nagato_train,
              args.nagato_validation, args.nagato_test, args.tokenizer]
    files = {
        "train": args.output_dir / "binding_train_v100.jsonl",
        "validation": args.output_dir / "binding_validation_v100.jsonl",
        "test": args.output_dir / "binding_test_v100.jsonl",
        "control": args.output_dir / "binding_test_no_evidence_v100.jsonl",
        "report": args.output_dir / "binding_report_v100.json",
    }
    if len({p.resolve() for p in inputs + list(files.values())}) != len(inputs) + len(files):
        raise ValueError("Output path overlaps input or another output")
    if not args.overwrite and any(p.exists() for p in files.values()):
        raise FileExistsError("Output exists; choose a fresh --output-dir")
    persona_train_all = load_dialogue_rows(args.persona_train, 0, args.seed)
    persona_val_all = load_dialogue_rows(args.persona_validation, 0, args.seed)
    # Unified v0.9.9 already oversamples Nagato. Drop those repeated rows
    # before constructing a fresh explicitly controlled mixture.
    persona_train = [r for r in persona_train_all
                     if r.get("source_kind", "real_persona_chat") == "real_persona_chat"]
    persona_val = [r for r in persona_val_all
                   if r.get("source_kind", "real_persona_chat") == "real_persona_chat"]
    if not persona_train or not persona_val:
        raise ValueError("Missing Persona rows")
    bound, stats = {}, {}
    for split, path in (("train", args.nagato_train),
                        ("validation", args.nagato_validation),
                        ("test", args.nagato_test)):
        input_rows = load_dialogue_rows(path, 0, args.seed)
        bound[split], stats[split] = prepare_bound(
            input_rows, tok, args.context_length, args.excerpt_chars, args.max_unk_fraction)
        if not bound[split]:
            raise ValueError("No usable binding rows in " + split)
    check_partitions(bound["train"], bound["validation"], bound["test"])
    if {r["conversation_id"] for r in persona_train} & {r["conversation_id"] for r in persona_val}:
        raise ValueError("Persona train and validation overlap")
    mixed_train = mix_training_rows(persona_train, bound["train"],
                                   nagato_fraction=args.nagato_fraction, seed=args.seed)
    mixed_val = [*persona_val, *bound["validation"]]
    if ({r["conversation_id"] for r in mixed_train}
            & {r["conversation_id"] for r in mixed_val}):
        raise ValueError("Unified source leakage")
    control = no_evidence(bound["test"], tok, args.context_length, args.max_unk_fraction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, rows in (("train", mixed_train), ("validation", mixed_val),
                      ("test", bound["test"]), ("control", control)):
        write_jsonl(files[key], rows)
    counts = Counter(r["source_kind"] for r in mixed_train)
    report = {
        "version": "v0.10.0", "train_total": len(mixed_train),
        "validation_total": len(mixed_val), "train_by_source": dict(counts),
        "binding_by_split": {s: len(rows) for s, rows in bound.items()},
        "ablation_test_pairs": len(control), "filtered_counts": stats,
        "nagato_fraction_actual": counts["nagato_corpus"] / len(mixed_train),
        "context_length": args.context_length,
        "limitation": (
            "Oracle evidence is in every bound input; this tests evidence-conditioned "
            "generation, not automatic retrieval or internal factual memory. Held-out "
            "source lines prevent exact-source reuse across splits but not conceptual "
            "overlap. Generated local data includes private conversations and must "
            "not be committed."),
    }
    files["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Evidence-conditioned Nagato+Persona SFT")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--persona-train", type=Path, default=Path(
        "data/dialogue/unified_v099/unified_train_v099.jsonl"))
    p.add_argument("--persona-validation", type=Path, default=Path(
        "data/dialogue/unified_v099/unified_validation_v099.jsonl"))
    p.add_argument("--nagato-train", type=Path, default=Path(
        "data/dialogue/unified_v099/nagato_train_v099.jsonl"))
    p.add_argument("--nagato-validation", type=Path, default=Path(
        "data/dialogue/unified_v099/nagato_validation_v099.jsonl"))
    p.add_argument("--nagato-test", type=Path, default=Path(
        "data/dialogue/unified_v099/nagato_test_v099.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("data/dialogue/binding_v100"))
    p.add_argument("--nagato-fraction", type=float, default=.05)
    p.add_argument("--excerpt-chars", type=int, default=56)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--max-unk-fraction", type=float, default=.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    generate(args)


if __name__ == "__main__":
    main()
