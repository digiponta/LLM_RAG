"""Convert local Real Persona Chat JSON dialogues into short dialogue SFT records.

Input data stay on the user's Windows computer; this script never uploads them.
No audit/approval system. Splits by source dialogue FILE, never by utterance.
No model training or changing of checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_SOURCE = Path("../LLM_CHA/data-src/real-persona-chat/real_persona_chat/dialogues")
SPLITS = ("train", "validation", "test")


def split_for(path: Path, seed: int, validation_fraction: float, test_fraction: float) -> str:
    """Deterministic split stable across traversal order."""
    digest = hashlib.sha256((str(seed) + ":" + path.as_posix()).encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / (2 ** 64)
    if fraction < test_fraction:
        return "test"
    if fraction < test_fraction + validation_fraction:
        return "validation"
    return "train"


def clean_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def dialogue_pairs(data: object, source: str, max_prompt_chars: int = 27,
                   max_answer_chars: int = 30, min_answer_chars: int = 2,
                   with_context: bool = False) -> list[dict]:
    """Extract *adjacent turns*, never merge utterances across speakers.

    Each speaker is used as a potential responding assistant, and AA/AB
    speaker IDs are preserved. Role identity is not treated as Nagato.
    """
    if not isinstance(data, dict) or not isinstance(data.get("utterances"), list):
        raise ValueError("Expected JSON object with utterances array")
    utterances = data["utterances"]
    examples = []
    for index in range(1, len(utterances)):
        before, current = utterances[index - 1], utterances[index]
        if not isinstance(before, dict) or not isinstance(current, dict):
            continue
        speaker_before, speaker_after = before.get("interlocutor_id"), current.get("interlocutor_id")
        if not speaker_before or not speaker_after or speaker_before == speaker_after:
            continue
        question, answer = clean_text(before.get("text")), clean_text(current.get("text"))
        if not question or not answer:
            continue
        if len(question) > max_prompt_chars or not min_answer_chars <= len(answer) <= max_answer_chars:
            continue
        if "�" in (question + answer):
            continue
        context = ""
        if with_context and index >= 2:
            previous = utterances[index - 2]
            if isinstance(previous, dict) and previous.get("interlocutor_id") == speaker_after:
                value = clean_text(previous.get("text"))
                if len(value) <= 15:
                    context = value
        examples.append({
            "id": f"{source}:{index}",
            "conversation_id": source,
            "source_utterance_id": current.get("utterance_id", index),
            "question": question,
            "answer": answer,
            "context": context,
            "question_speaker": str(speaker_before),
            "answer_speaker": str(speaker_after),
            "example_type": "real_dialogue_adjacent_turn",
        })
    return examples


def compact_prompt(row: dict) -> str:
    context = row.get("context")
    return (f"前:{context}\n" if context else "") + f"問:{row['question']}\n答:"


def prepare(source_dir: Path, output_dir: Path, max_context: int = 64,
            min_answer_chars: int = 2, with_context: bool = False,
            seed: int = 42, validation_fraction: float = .1,
            test_fraction: float = .1, overwrite: bool = False) -> dict:
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if not 0 < validation_fraction < .5 or not 0 < test_fraction < .5:
        raise ValueError("validation/test fractions must be in (0, 0.5)")
    if validation_fraction + test_fraction >= .75:
        raise ValueError("train fraction too small")
    if max_context < 20:
        raise ValueError("max_context must be >=20")

    dests = {name: output_dir / f"real_persona_{name}_v081.jsonl" for name in SPLITS}
    report_path = output_dir / "real_persona_report_v081.json"
    for path in [*dests.values(), report_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Existing output: {path}; use --overwrite to replace")

    input_files = sorted(source_dir.rglob("*.json"))
    if not input_files:
        raise ValueError("No *.json files found")

    counters = Counter()
    rows = {name: [] for name in SPLITS}
    hashes = set()
    files_per_split = Counter()
    for file in input_files:
        source = file.relative_to(source_dir).as_posix()
        split = split_for(Path(source), seed, validation_fraction, test_fraction)
        files_per_split[split] += 1
        try:
            data = json.loads(file.read_text(encoding="utf-8-sig"))
            pairs = dialogue_pairs(
                data, source, max_prompt_chars=max_context // 2,
                max_answer_chars=max_context // 2,
                min_answer_chars=min_answer_chars, with_context=with_context,
            )
        except (UnicodeError, ValueError, OSError, json.JSONDecodeError):
            counters["invalid_json_or_schema"] += 1
            continue
        counters["parsed_files"] += 1
        counters["candidate_pairs"] += len(pairs)
        for row in pairs:
            # Note +1 for BOS; a target token after the last answer character
            # needs room for EOS. Character tokenizer -> one char per token.
            if len(compact_prompt(row)) + len(row["answer"]) + 2 > max_context:
                counters["over_context_limit"] += 1
                continue
            fingerprint = hashlib.sha256(
                (row["question"] + "\u241f" + row["answer"] + "\u241f" + row["context"]).encode("utf-8")
            ).hexdigest()
            if fingerprint in hashes:
                counters["duplicate_pairs"] += 1
                continue
            hashes.add(fingerprint)
            rows[split].append(row)

    if any(not rows[name] for name in SPLITS):
        raise ValueError("One or more splits have no records; check source or filtering thresholds")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        with dests[name].open("w", encoding="utf-8") as dest:
            for row in rows[name]:
                dest.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "input_files": len(input_files),
        "files_per_split": dict(files_per_split),
        "records_per_split": {name: len(rows[name]) for name in SPLITS},
        "counts": dict(counters),
        "max_context_chars": max_context,
        "with_context": with_context,
        "split_unit": "source dialogue file",
        "deduplication": "exact question/answer/context duplicate dropped globally",
        "limitations": ("Character length is a proxy for the compatible LLM_GPU character tokenizer; "
                        "check unknown-token rate when training. No semantic/persona quality checks; "
                        "conversation-level split is not enough to rule out near-duplicate or participant leakage."),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Prepare locally stored Real Persona Chat data for LLM_RAG")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=Path("data/dialogue/real_persona_v081"))
    parser.add_argument("--max-context", type=int, default=64)
    parser.add_argument("--with-context", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--test-fraction", type=float, default=.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = prepare(args.input_dir, args.output_dir,
                     max_context=args.max_context, with_context=args.with_context,
                     seed=args.seed, validation_fraction=args.validation_fraction,
                     test_fraction=args.test_fraction, overwrite=args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("NOTE: No model training or remote upload was performed.")


if __name__ == "__main__":
    main()
