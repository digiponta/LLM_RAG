"""v0.9.2: matched legacy versus speaker/turn-structured prompts.

Use exactly the same question/answer IDs and existing dialogue-file splits.
Original private JSON remains local; no source material uploaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from prepare_context_dialogue_v087 import context_for, previous_turns
from prepare_real_persona_v081 import DEFAULT_SOURCE, SPLITS, clean_text
from train_real_dialogue_v082 import encode_dialogue, load_dialogue_rows


def structured_prompt(question: str, question_speaker: str, answer_speaker: str,
                      history: list[tuple[str, str]]) -> str:
    """A always designates respondent, B the questioner, relative to each row.

    A/B are NOT original real-world identity labels. Turn order is indicated
    by newline order and explicit speaker prefix; no positional model change.
    """
    if not question or not question_speaker or not answer_speaker or question_speaker == answer_speaker:
        raise ValueError("Missing/invalid speaker roles")
    labels = {answer_speaker: "A", question_speaker: "B"}
    result = []
    for speaker, text in history:
        if speaker not in labels or not text:
            raise ValueError("Invalid historical turn")
        result.append(f"{labels[speaker]}:{text}")
    result.extend([f"B:{question}", "A:"])
    return "\n".join(result)


def reconstruct(base: dict, dialogue: dict, turns: int) -> dict:
    utterances = dialogue.get("utterances")
    try:
        index = int(base["id"].rsplit(":", 1)[1])
    except (ValueError, KeyError, IndexError) as exc:
        raise ValueError("Invalid original row identifier") from exc
    if not isinstance(utterances, list) or index < 2 or index >= len(utterances):
        raise ValueError("Source question/answer not present")
    current, question = utterances[index], utterances[index - 1]
    if not isinstance(current, dict) or not isinstance(question, dict):
        raise ValueError("Invalid utterance schema")
    if (clean_text(current.get("text")) != base["answer"]
        or clean_text(question.get("text")) != base["question"]
        or current.get("interlocutor_id") != base.get("answer_speaker")
        or question.get("interlocutor_id") != base.get("question_speaker")):
        raise ValueError("Original source and QA metadata mismatch")
    history_text = context_for(dialogue, index, turns)
    if not history_text or history_text != base.get("context"):
        raise ValueError("Original source and context differ")
    values = previous_turns(dialogue, index, turns)
    actual = []
    # preceding turns are contiguous, ordered oldest to newest, ending at index-2
    start = index - 1 - len(values)
    for i, value in enumerate(values):
        utterance = utterances[start + i]
        actual.append((utterance["interlocutor_id"], value))
    override = structured_prompt(base["question"], base["question_speaker"],
                                 base["answer_speaker"], actual)
    return dict(base, prompt_override=override)


def paired_rows(rows: list[dict], root: Path, tok, length: int, turns: int) -> tuple[list[dict],list[dict],dict]:
    legacy, structured = [], []
    counts = Counter()
    cache_id, cache_data = None, None
    root = root.resolve()
    for base in rows:
        relative = base["conversation_id"]
        source = (root / relative).resolve()
        if root not in source.parents:
            raise ValueError("Source path outside source root")
        if cache_id != relative:
            cache_data = json.loads(source.read_text(encoding="utf-8-sig"))
            cache_id = relative
        try:
            modified = reconstruct(base, cache_data, turns)
        except ValueError:
            counts["invalid_original_or_history"] += 1
            continue
        for candidate, label in ((base, "legacy"), (modified, "structured")):
            _, status = encode_dialogue(candidate, tok, length)
            if status != "accepted":
                counts[label + "_" + status] += 1
                break
        else:
            legacy.append(dict(base))
            structured.append(modified)
            counts["paired_accepted"] += 1
    return legacy, structured, dict(counts)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(args):
    if args.context_length < 32 or args.turns not in (1, 2):
        raise ValueError("Invalid context length or turn count")
    outputs = {
        (split, variant): args.output_dir / f"paired_{variant}_{split}_v092.jsonl"
        for split in SPLITS for variant in ("legacy", "structured")
    }
    report_file = args.output_dir / "paired_report_v092.json"
    for path in [*outputs.values(), report_file]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Existing output: {path}")
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    from tokenizer import Tokenizer
    tokenizer = Tokenizer.load(str(args.tokenizer))
    stats = {}
    prepared = {}
    seen_ids = set()
    seen_conversations = set()
    for split in SPLITS:
        input_file = args.base_dir / f"paired_context_{split}_v087.jsonl"
        rows = load_dialogue_rows(input_file, 0, args.seed)
        legacy, structured, counts = paired_rows(
            rows, args.source_dir, tokenizer, args.context_length, args.turns)
        ids = {row["id"] for row in legacy}
        convs = {row["conversation_id"] for row in legacy}
        if ids & seen_ids or convs & seen_conversations:
            raise ValueError("Cross-split leakage")
        if [r["id"] for r in legacy] != [r["id"] for r in structured]:
            raise AssertionError("Mismatched paired records")
        if not legacy:
            raise ValueError(f"No usable pairs in {split}")
        seen_ids.update(ids)
        seen_conversations.update(convs)
        prepared[split] = {"legacy": legacy, "structured": structured}
        stats[split] = {"input": len(rows), "paired": len(legacy), "reasons": counts}
        print(f"{split}: {len(legacy)}/{len(rows)}, {counts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (split, variant), filename in outputs.items():
        write_jsonl(filename, prepared[split][variant])
    report = {
        "version": "v0.9.2", "context_length": args.context_length,
        "turns": args.turns, "splits": stats,
        "source": "v0.9.0 dialogue-file-separated paired_context datasets",
        "method": "Identical QA IDs, full original answer and question preserved; accept only if both encodings fit and pass UNK filter.",
        "prompt_style": "A=responder, B=questioner (roles relative per example); chronological history prefixed A:/B:, followed by B:question\\nA:.",
        "limitations": "Original short-turn corpus only; position encoding not added. Rebuilding from source is required for correct history-speaker labels. Source and output JSONL are private, keep local.",
    }
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Report:", report_file)
    return report


def main():
    p = argparse.ArgumentParser(description="Build paired legacy/role-structured Real Persona Chat SFT inputs")
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--base-dir", type=Path, default=Path("data/dialogue/real_persona_context256_v090"))
    p.add_argument("--output-dir", type=Path, default=Path("data/dialogue/real_persona_structured_v092"))
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--turns", type=int, choices=(1, 2), default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
