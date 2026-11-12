"""v0.8.7: restore real preceding turns for PAIRED context/no-context SFT.

Works exclusively on local Real Persona Chat JSON and the existing v0.8.1
dialogue-file-separated splits. Does not alter the prior dataset/checkpoints.
No fabricated prior turns; no truncation of question/answer/context.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_real_persona_v081 import DEFAULT_SOURCE, SPLITS, clean_text, compact_prompt


def previous_turns(dialogue: dict, answer_index: int, max_turns: int = 2) -> list[str]:
    """Return earliest-to-latest immediately preceding history, excluding question.

    Require preceding turn's speaker to be the ANSWER speaker, so that
    abbreviated questions like 'あなたは？' have role-consistent context.
    Previous turns must be contiguous valid utterances; never stitch gaps.
    """
    utterances = dialogue.get("utterances")
    if not isinstance(utterances, list) or answer_index < 2 or max_turns < 1:
        return []
    if answer_index >= len(utterances):
        return []
    answer = utterances[answer_index]
    if not isinstance(answer, dict) or not answer.get("interlocutor_id"):
        return []
    reply_speaker = answer["interlocutor_id"]
    if not isinstance(utterances[answer_index - 2], dict):
        return []
    if utterances[answer_index - 2].get("interlocutor_id") != reply_speaker:
        return []
    values = []
    for index in range(answer_index - 2, max(-1, answer_index - 2 - max_turns), -1):
        row = utterances[index]
        if not isinstance(row, dict) or not row.get("interlocutor_id"):
            break
        text = clean_text(row.get("text"))
        if not text or "�" in text:
            break
        values.append(text)
    return list(reversed(values))


def context_for(dialogue: dict, answer_index: int, turns: int) -> str:
    """Join authentic chronological history with visible turn separators."""
    return "／".join(previous_turns(dialogue, answer_index, turns))


def fit_tokens(row: dict, tokenizer, context_limit: int, max_unk_fraction: float) -> str:
    from train_real_dialogue_v082 import encode_dialogue
    _, status = encode_dialogue(row, tokenizer, context_limit, max_unk_fraction)
    return status


def build_paired(rows: list[dict], source_dir: Path, tokenizer, context_limit: int,
                 turns: int, max_unk_fraction: float) -> tuple[list[dict], list[dict], dict]:
    """Only retain examples that fit identically with and without context."""
    plain, conditioned = [], []
    counts = Counter()
    cache_name = None
    cache_dialogue = None
    root = source_dir.resolve()
    for row in rows:
        name = row["conversation_id"]
        source = (root / name).resolve()
        if source != root and root not in source.parents:
            raise ValueError("Conversation path escapes source directory")
        if cache_name != name:
            cache_dialogue = json.loads(source.read_text(encoding="utf-8-sig"))
            cache_name = name
        try:
            answer_index = int(row["id"].rsplit(":", 1)[1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Unrecognized v0.8.1 row ID: {row['id']}") from exc
        utterances = cache_dialogue.get("utterances")
        if not isinstance(utterances, list) or answer_index >= len(utterances):
            counts["source_row_missing"] += 1
            continue
        answer = utterances[answer_index]
        question = utterances[answer_index - 1]
        if not isinstance(answer, dict) or not isinstance(question, dict):
            counts["source_row_invalid"] += 1
            continue
        if clean_text(answer.get("text")) != row["answer"] or clean_text(question.get("text")) != row["question"]:
            counts["source_row_changed"] += 1
            continue
        history = context_for(cache_dialogue, answer_index, turns)
        if not history:
            counts["no_role_consistent_history"] += 1
            continue
        base = dict(row, context="")
        contextual = dict(row, context=history)
        baseline_status = fit_tokens(base, tokenizer, context_limit, max_unk_fraction)
        context_status = fit_tokens(contextual, tokenizer, context_limit, max_unk_fraction)
        if baseline_status != "accepted":
            counts["baseline_" + baseline_status] += 1
            continue
        if context_status != "accepted":
            counts["context_" + context_status] += 1
            continue
        plain.append(base)
        conditioned.append(contextual)
        counts["paired_accepted"] += 1
        counts["context_chars_total"] += len(history)
        counts["context_turns_" + str(len(previous_turns(cache_dialogue, answer_index, turns)))] += 1
    return plain, conditioned, dict(counts)


def output_files(output_dir: Path) -> dict:
    files = {}
    for split in SPLITS:
        for variant in ("plain", "context"):
            files[(split, variant)] = output_dir / f"paired_{variant}_{split}_v087.jsonl"
    return files


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args) -> dict:
    if args.context_length < 20 or args.turns not in (1, 2):
        raise ValueError("context-length must be >=20; turns must be 1 or 2")
    if not 0 <= args.max_unk_fraction <= 1:
        raise ValueError("Invalid max-unk-fraction")
    from train_real_dialogue_v082 import load_dialogue_rows
    paths = output_files(args.output_dir)
    report_path = args.output_dir / "context_report_v087.json"
    for dest in [*paths.values(), report_path]:
        if dest.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {dest}")
    import sys
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from tokenizer import Tokenizer
    tok = Tokenizer.load(str(args.tokenizer))
    stats = {}
    paired = {}
    seen = {}
    for split in SPLITS:
        filename = args.base_dir / f"real_persona_{split}_v081.jsonl"
        base_rows = load_dialogue_rows(filename, 0, 42)
        plain, context, counts = build_paired(base_rows, args.source_dir, tok,
                                               args.context_length, args.turns,
                                               args.max_unk_fraction)
        ids_plain = [r["id"] for r in plain]
        if ids_plain != [r["id"] for r in context]:
            raise AssertionError("Unpaired records")
        ids = {r["id"] for r in plain}
        conversations = {r["conversation_id"] for r in plain}
        if any(ids.intersection(group["ids"]) or conversations.intersection(group["conversations"]) for group in seen.values()):
            raise ValueError("Data leakage between splits")
        seen[split] = {"ids": ids, "conversations": conversations}
        stats[split] = {"original_rows": len(base_rows), "paired_rows": len(plain), "counts": counts}
        paired[split] = {"plain": plain, "context": context}
        print(f"{split}: {len(plain)}/{len(base_rows)} paired, reasons={counts}")
    if any(not paired[split]["context"] for split in SPLITS):
        raise ValueError("One or more split has zero usable context pairs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (split, variant), dest in paths.items():
        write_rows(dest, paired[split][variant])
    report = {
        "version": "v0.8.7", "task": "paired context ablation",
        "context_length": args.context_length, "history_turns": args.turns,
        "original_split": "v0.8.1 dialogue-file split (same conversation_id mapping)",
        "splits": stats, "outputs": {f"{s}_{v}": str(p) for (s, v), p in paths.items()},
        "limitations": (
            "Paired cohort is a SELECTED subset that fits full context under the existing model. "
            "Reported rates cannot be generalized to full real-dialogue corpus. "
            "Both variants preserve the same question/answer and same dialogue-file split. "
            "A larger context model is required to measure longer histories without exclusion. "
            "Source data and generated JSONL contain private conversation text; keep local."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Report:", report_path)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare paired no-context/context v0.8.7 datasets")
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--base-dir", type=Path, default=Path("data/dialogue/real_persona_v081"))
    p.add_argument("--output-dir", type=Path, default=Path("data/dialogue/real_persona_context_v087"))
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--context-length", type=int, default=64)
    p.add_argument("--turns", type=int, choices=(1, 2), default=2)
    p.add_argument("--max-unk-fraction", type=float, default=.1)
    p.add_argument("--overwrite", action="store_true")
    build(p.parse_args())


if __name__ == "__main__":
    main()
