"""LLM_RAG v0.9.9 — one SFT interface for Real Persona Chat and Nagato corpus.

Both sources become rows accepted WITHOUT MODIFICATION by
train_real_dialogue_v082.py: id, conversation_id, question, answer, context.
Nagato text is deliberately attributed as source statements, not independent
verified scientific facts. Keep all generated Real Persona Chat mixtures local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from prepare_real_persona_v081 import compact_prompt
from train_real_dialogue_v082 import load_dialogue_rows

TOPICS = (
    "宇宙", "時間", "空間", "量子", "原子", "粒子", "物理", "数学", "論理",
    "知識", "情報", "プログラム", "人間", "脳", "意識", "言語", "社会",
    "映画", "読書", "本", "仕事", "科学", "重力", "光", "電場", "磁場",
)


def split_for_line(line: str, *, seed: int = 42,
                   validation_fraction: float = .1, test_fraction: float = .1) -> str:
    """Assign an ORIGINAL LINE to one split; all its chunks stay together."""
    if not (0 <= validation_fraction < 1 and 0 <= test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError("Invalid fractions")
    fingerprint = hashlib.sha256((str(seed) + ":" + line).encode("utf-8")).digest()
    value = int.from_bytes(fingerprint[:8], "big") / 2**64
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def segment_text(text: str, max_chars: int = 100) -> list[str]:
    """Break lengthy statements at punctuation when possible, never invent words."""
    if max_chars < 24:
        raise ValueError("max_chars too small")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    rest = text
    while rest:
        if len(rest) <= max_chars:
            chunks.append(rest)
            break
        cut = max(
            (m.end() for m in re.finditer(r"[。！？!?、，；;]", rest[:max_chars])
             if m.end() >= max_chars // 2),
            default=max_chars,
        )
        chunks.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    return [x for x in chunks if x]


def make_nagato_rows(text: str, *, seed: int = 42,
                     max_chars: int = 90, context_length: int = 256,
                     tokenizer=None) -> dict[str, list[dict]]:
    """Faithful attributed-QA AND authored-persona-response examples from corpus.

    No hallucinated paraphrases or fabricated Q/A facts. Unknown segments are
    not silently discarded: the caller reports all rejects.
    """
    splits = {"train": [], "validation": [], "test": []}
    seen = set()
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("@NagatFakeBot") or line.startswith("#"):
            continue
        digest = hashlib.sha256(line.encode("utf-8")).hexdigest()[:16]
        if digest in seen:
            continue
        seen.add(digest)
        split = split_for_line(line, seed=seed)
        for index, chunk in enumerate(segment_text(line, max_chars=max_chars)):
            topic = next((topic for topic in TOPICS if topic in chunk), None)
            if topic is None:
                # A short exact excerpt is a retrieval cue, not an invented title.
                topic = chunk[:min(12, len(chunk))]
            # Each source statement needs a distinct prompt. A broad topic-only
            # question would map many contradictory statements to one answer,
            # encouraging generic replies instead of learning the source.
            cue = chunk[:min(20, len(chunk))]
            question = f"長門コーパスで「{topic}」について「{cue}」と始まる記述は？"
            answer = f"長門コーパスには「{chunk}」と記されています。"
            row = {
                "id": f"nagato-v099:{digest}:{index}",
                "conversation_id": f"nagato-v099:{digest}",
                "question": question,
                "answer": answer,
                "context": "",
                "source_kind": "nagato_corpus",
                "source_line": line_no,
                "original_line_hash": digest,
                "review_state": "SOURCE_QUOTE_UNVERIFIED",
                "example_type": "source_attributed_qa",
            }
            persona_row = dict(row)
            persona_row["id"] = row["id"] + ":persona"
            persona_row["question"] = f"長門として「{topic}」について、次の書き出しに沿って話してください：「{cue}」"
            persona_row["answer"] = chunk
            persona_row["example_type"] = "persona_style_from_source"
            row["id"] += ":source"
            for variant in (row, persona_row):
                if tokenizer is not None:
                    prefix = tokenizer.encode(compact_prompt(variant), add_bos=True)
                    targets = tokenizer.encode(variant["answer"])
                    if len(prefix) + len(targets) > context_length:
                        continue
                splits[split].append(variant)
    return splits


def check_splits(splits: dict[str, list[dict]]):
    ids, origins = set(), {}
    for split, rows in splits.items():
        for row in rows:
            if row["id"] in ids:
                raise ValueError("Duplicate ID across splits")
            ids.add(row["id"])
            source = row["conversation_id"]
            if source in origins and origins[source] != split:
                raise ValueError("Source line leaked between splits")
            origins[source] = split


def mix_training_rows(persona_rows: list[dict], nagato_rows: list[dict],
                      *, nagato_fraction: float = .05,
                      seed: int = 42) -> list[dict]:
    """Oversample only the Nagato TRAIN partition to a target fraction.

    Repeated rows have unique sample IDs but the SAME conversation_id, keeping
    their provenance auditable. Validation and test are NEVER oversampled.
    """
    if not (0 < nagato_fraction < .5):
        raise ValueError("nagato_fraction must be in (0, 0.5)")
    if not persona_rows or not nagato_rows:
        raise ValueError("Both training sources must contain examples")
    goal = max(1, round(len(persona_rows) * nagato_fraction / (1 - nagato_fraction)))
    rand = random.Random(seed)
    sequence = list(nagato_rows)
    rand.shuffle(sequence)
    selected = [sequence[i % len(sequence)] for i in range(goal)]
    mixed = [{**row, "source_kind": row.get("source_kind", "real_persona_chat")}
             for row in persona_rows]
    for i, row in enumerate(selected):
        copy = dict(row)
        # The mixed SFT schema must be source-complete even if a caller
        # supplies minimal Nagato rows without optional metadata.
        copy["source_kind"] = "nagato_corpus"
        copy["id"] = f"{row['id']}:repeat{i}"
        copy["repeat_index"] = i
        mixed.append(copy)
    rand.shuffle(mixed)
    return mixed


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args):
    import sys
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from tokenizer import Tokenizer
    if args.output_dir.resolve() in (args.nagato.resolve(), args.persona_train.resolve(),
                                      args.persona_validation.resolve()):
        raise ValueError("Output dir cannot be an input file")
    tok = Tokenizer.load(str(args.tokenizer))
    corpus = args.nagato.read_text(encoding="utf-8-sig")
    source_splits = make_nagato_rows(corpus, seed=args.seed,
                                     max_chars=args.max_chars,
                                     context_length=args.context_length,
                                     tokenizer=tok)
    check_splits(source_splits)
    if any(not source_splits[split] for split in source_splits):
        raise ValueError("Nagato split empty; use a sufficiently large corpus")
    persona_train = load_dialogue_rows(args.persona_train, 0, args.seed)
    persona_validation = load_dialogue_rows(args.persona_validation, 0, args.seed)
    persona_train_origins = {r["conversation_id"] for r in persona_train}
    persona_val_origins = {r["conversation_id"] for r in persona_validation}
    if persona_train_origins & persona_val_origins:
        raise ValueError("Persona train/validation source overlap")
    if {r["id"] for r in persona_train} & {r["id"] for r in persona_validation}:
        raise ValueError("Persona train/validation ID overlap")
    outputs = {
        "train": args.output_dir / "unified_train_v099.jsonl",
        "validation": args.output_dir / "unified_validation_v099.jsonl",
        "nagato_test": args.output_dir / "nagato_test_v099.jsonl",
        "nagato_train": args.output_dir / "nagato_train_v099.jsonl",
        "nagato_validation": args.output_dir / "nagato_validation_v099.jsonl",
        "report": args.output_dir / "unified_report_v099.json",
    }
    if not args.overwrite:
        for p in outputs.values():
            if p.exists():
                raise FileExistsError(f"Already exists: {p}")
    train = mix_training_rows(persona_train, source_splits["train"],
                              nagato_fraction=args.nagato_fraction, seed=args.seed)
    validation = [
        *({**row, "source_kind": row.get("source_kind", "real_persona_chat")}
          for row in persona_validation),
        *source_splits["validation"],
    ]
    if {r["id"] for r in train} & {r["id"] for r in validation}:
        raise ValueError("Train/validation row overlap")
    if {r["conversation_id"] for r in train} & {r["conversation_id"] for r in validation}:
        raise ValueError("Train/validation source overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(outputs["train"], train)
    write_jsonl(outputs["validation"], validation)
    for split in source_splits:
        write_jsonl(outputs[f"nagato_{split}"], source_splits[split])
    counts = Counter(row["source_kind"] for row in train)
    report = {
        "version": "v0.9.9",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_by_source": dict(counts),
        "nagato_splits": {name: len(rows) for name, rows in source_splits.items()},
        "requested_nagato_train_fraction": args.nagato_fraction,
        "actual_nagato_train_fraction": counts["nagato_corpus"] / len(train),
        "nagato_original_lines": sum(bool(x.strip()) for x in corpus.splitlines()),
        "tokenizer_vocab": tok.vocab_size,
        "context_length": args.context_length,
        "qa_format": "compact_prompt: optional 前:context / 問:question / 答:answer",
        "loss": "masked answer-token causal cross entropy (train_real_dialogue_v082.py)",
        "provenance": "SOURCE_QUOTE_UNVERIFIED; corpus statements may include fiction, hypotheses and opinions",
        "warnings": (
            "Nagato examples are *synthetic source-attributed questions*, not genuine chat turns. "
            "Heavy oversampling may cause memorization and catastrophic forgetting. "
            "The historical Real Persona Chat test split must remain untouched. "
            "Review original corpus permissions before redistribution. "
            "No raw private Persona Chat text appears in report; generated JSONLs are LOCAL ONLY."),
    }
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Prepare one SFT dataset from Persona and Nagato")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--nagato", type=Path, default=Path("data/knowledge/data-nagato.txt"))
    p.add_argument("--persona-train", type=Path, default=Path(
        "data/dialogue/real_persona_structured_v092/paired_legacy_train_v092.jsonl"))
    p.add_argument("--persona-validation", type=Path, default=Path(
        "data/dialogue/real_persona_structured_v092/paired_legacy_validation_v092.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("data/dialogue/unified_v099"))
    p.add_argument("--nagato-fraction", type=float, default=.05)
    p.add_argument("--max-chars", type=int, default=90)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
