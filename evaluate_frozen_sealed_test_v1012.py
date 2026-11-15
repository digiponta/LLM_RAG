"""v0.10.12: frozen checkpoint preflight and ONE-SHOT sealed TEST evaluation.

Preflight never opens Test files. Explicit --open-final-test is required to
calculate scores; results are local and original Test data is read-only.
This tests existing source-separated corpus and optional Persona Test splits,
not unseen semantic concepts, natural conversation quality or retrieval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from evaluate_nagato_retention_v106 import choose_rows
from train_real_dialogue_v082 import (
    collate_dialogue, encode_dialogue, evaluate_answer_nll, load_dialogue_rows)
from prepare_real_persona_v081 import compact_prompt

TEST_TASKS = ("nagato_knowledge", "nagato_persona",
              "real_persona", "evidence_binding")
DEFAULT_CHECKPOINT = Path("model/model-rag-position-rehearsal-v0.10.11.pt")


def sha256_file(path, block=1 << 20):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            data = stream.read(block)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def origin(row):
    """Underlying original Nagato line or Persona dialogue, not derived sample ID."""
    v = row.get("original_conversation_id", row["conversation_id"])
    for _ in range(6):
        prefixes = ("balanced-v107:train:", "balanced-v107:validation:",
                    "position-v105:train:", "position-v105:development:",
                    "copy-v103:train:", "copy-v103:development:")
        matched = next((p for p in prefixes if v.startswith(p)), None)
        if matched is None:
            break
        v = v[len(matched):]
    return v


def audit_partition(train, heldout):
    """Reject cross-partition source IDs, row IDs and exact Q/A duplicates."""
    train_origins = {origin(r) for r in train}
    train_ids = {r["id"] for r in train}
    train_qa = {
        (r["question"], r["answer"], r.get("context", ""))
        for r in train
    }
    if len({r["id"] for r in heldout}) != len(heldout):
        raise ValueError("Duplicate IDs inside sealed TEST")
    if any(origin(r) in train_origins for r in heldout):
        raise ValueError("TRAIN / final TEST origin overlap")
    if any(r["id"] in train_ids for r in heldout):
        raise ValueError("TRAIN / final TEST row ID overlap")
    if any((r["question"], r["answer"], r.get("context", "")) in train_qa
           for r in heldout):
        raise ValueError("TRAIN / final TEST exact Q/A overlap")
    return {
        "train_rows": len(train),
        "test_rows": len(heldout),
        "origin_overlap": 0,
        "row_id_overlap": 0,
        "exact_qa_overlap": 0,
    }


def check_inputs(args):
    if args.max_rows < 0 or args.generations < 0 or args.batch_size < 1:
        raise ValueError("Invalid evaluation arguments")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if not 0 <= args.max_unk_fraction <= 1:
        raise ValueError("Invalid unknown threshold")
    if args.report.resolve() in {p.resolve() for p in [
        args.checkpoint, args.baseline, args.nagato_test, args.binding_test,
        args.tokenizer, args.train_mixed, args.persona_test
    ] if p is not None}:
        raise ValueError("Report overwrites an input")
    if args.report.exists():
        raise FileExistsError("Report already exists; sealed TEST cannot be rerun into same artifact")
    if args.checkpoint.resolve() == args.baseline.resolve():
        raise ValueError("Candidate and baseline must be distinct checkpoints")
    for p in (args.checkpoint, args.baseline, args.tokenizer, args.train_mixed):
        if not p.is_file():
            raise FileNotFoundError(p)


def preflight(args):
    """Does not open, stat, or hash any TEST file."""
    check_inputs(args)
    return {
        "version": "v0.10.12",
        "status": "READY_TO_OPEN_SEALED_TEST",
        "opened_final_test": False,
        "candidate": str(args.checkpoint),
        "candidate_sha256": sha256_file(args.checkpoint),
        "baseline": str(args.baseline),
        "baseline_sha256": sha256_file(args.baseline),
        "requested_test_paths_not_opened": [
            str(args.nagato_test), str(args.binding_test),
            *(str(args.persona_test),) if args.persona_test is not None else ()
        ],
        "note": "One-shot evaluation; no model or tokenizer is changed. "
                "Real Persona TEST must be provided explicitly if available. "
                "No Copy/Position independent TEST is claimed.",
    }


def _task_rows(nagato, binding, persona):
    result = {
        "nagato_knowledge": [
            x for x in nagato if x.get("example_type") == "source_attributed_qa"
            and x.get("source_kind") == "nagato_corpus"],
        "nagato_persona": [
            x for x in nagato if x.get("example_type") == "persona_style_from_source"
            and x.get("source_kind") == "nagato_corpus"],
        "evidence_binding": [
            x for x in binding if x.get("example_type") == "evidence_direct"
            and x.get("source_kind") == "nagato_corpus"],
    }
    if persona is not None:
        result["real_persona"] = [
            x for x in persona if x.get("source_kind", "real_persona_chat")
            == "real_persona_chat"]
    return result


def evaluate(args):
    import torch
    from torch.utils.data import DataLoader
    from tokenizer import Tokenizer
    from mha_rope_v096 import load_model_v096

    check_inputs(args)
    if not args.open_final_test:
        return preflight(args)

    # Never silently select a subset of the sealed TEST. Caller supplies a cap
    # deliberately, 0 evaluates every eligible example.
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    tok = Tokenizer.load(str(args.tokenizer))
    trained = load_dialogue_rows(args.train_mixed, 0, args.seed)
    nagato = load_dialogue_rows(args.nagato_test, 0, args.seed)
    binding = load_dialogue_rows(args.binding_test, 0, args.seed)
    persona = (load_dialogue_rows(args.persona_test, 0, args.seed)
               if args.persona_test is not None else None)

    audit = {
        "nagato": audit_partition(trained, nagato),
        "binding": audit_partition(trained, binding),
    }
    if persona is not None:
        audit["real_persona"] = audit_partition(trained, persona)
    # Additional same-original partition guards on the supplied sealed input.
    # Binding/Nagato may share an origin IN TEST, which is intentional.
    if {origin(r) for r in nagato} != {origin(r) for r in binding}:
        # They can differ if bound tokenizer filter excluded some original lines;
        # the strict prohibition is only TRAIN/TEST overlap, not equal coverage.
        pass

    models = {}
    for key, p in (("baseline_v100", args.baseline), ("candidate_v1011", args.checkpoint)):
        mdl, _ = load_model_v096(p, device, expected="rope")
        if mdl.vocab_size != tok.vocab_size:
            raise ValueError("Tokenizer and checkpoint vocab mismatch")
        models[key] = mdl.eval()

    task_data = _task_rows(nagato, binding, persona)
    if any(not group for group in task_data.values()):
        raise ValueError("At least one sealed TEST task group is empty")
    report = {
        "version": "v0.10.12",
        "status": "SEALED_TEST_EVALUATED",
        "opened_final_test": True,
        "validation_only": False,
        "accepted_for_deployment": False,
        "candidate_sha256": sha256_file(args.checkpoint),
        "baseline_sha256": sha256_file(args.baseline),
        "test_sha256": {
            "nagato": sha256_file(args.nagato_test),
            "binding": sha256_file(args.binding_test),
            **({"persona": sha256_file(args.persona_test)}
               if args.persona_test is not None else {})
        },
        "train_origin_audit": audit,
        "tasks": {},
        "limitations": (
            "Corpus TEST source lines were held out by deterministic line-hash split, "
            "but source facts may be unverified or fictional. Evidence is supplied "
            "as oracle, not retrieved. This report does not assess multi-turn chat, "
            "manual persona judgement, independent Position or Copy generalization. "
            "No output may be used to iteratively tune on this sealed TEST."
        ),
    }
    for task, all_rows in task_data.items():
        ordered = sorted(all_rows, key=lambda x: x["id"])
        if args.max_rows:
            ordered = ordered[:args.max_rows]
        accepted, pairs, rejects = [], [], Counter()
        context_limit = min(m.context_length for m in models.values())
        for row in ordered:
            pair, state = encode_dialogue(row, tok, context_limit,
                                           args.max_unk_fraction)
            if pair is None:
                rejects[state] += 1
            else:
                accepted.append(row)
                pairs.append(pair)
        if not accepted:
            raise ValueError(f"No valid TEST rows for {task}")
        loader = DataLoader(
            pairs, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda x: collate_dialogue(x, tok.pad_id))
        task_report = {
            "eligible": len(all_rows), "selected": len(ordered),
            "usable": len(accepted), "rejects": dict(rejects),
            "models": {},
        }
        for name, model in models.items():
            nll = evaluate_answer_nll(model, loader, device)
            samples = []
            for row in accepted[:args.generations]:
                prompt = tok.encode(compact_prompt(row), add_bos=True)
                with torch.inference_mode():
                    output = model.generate(
                        prompt, max_new_tokens=args.max_new_tokens,
                        eos_id=tok.eos_id, temperature=0,
                        top_k=None, repetition_penalty=1.0)
                suffix = output[len(prompt):]
                answer = tok.decode(suffix, skip_special_tokens=True)
                samples.append({
                    "exact": answer == row["answer"],
                    "eos": bool(suffix and suffix[-1] == tok.eos_id),
                    "output_chars": len(answer),
                    "expected_chars": len(row["answer"]),
                })
            task_report["models"][name] = {
                "answer_nll": nll,
                "generated_samples": len(samples),
                "exact_rate": sum(s["exact"] for s in samples) / len(samples)
                              if samples else None,
                "eos_rate": sum(s["eos"] for s in samples) / len(samples)
                            if samples else None,
                "mean_output_chars": sum(s["output_chars"] for s in samples) / len(samples)
                                     if samples else None,
            }
        report["tasks"][task] = task_report
        print(json.dumps({"task": task, "result": task_report}, ensure_ascii=False))
    # Persist once. No token sequences or private texts are logged.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    print(json.dumps({"status": report["status"], "report": str(args.report)}, ensure_ascii=False))
    return report


def main():
    p = argparse.ArgumentParser(description="Frozen v0.10.11 sealed Test gate; opt-in only")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--baseline", type=Path, default=Path(
        "model/model-rag-binding-v0.10.0.pt"))
    p.add_argument("--nagato-test", type=Path, default=Path(
        "data/dialogue/unified_v099/nagato_test_v099.jsonl"))
    p.add_argument("--binding-test", type=Path, default=Path(
        "data/dialogue/binding_v100/binding_test_v100.jsonl"))
    p.add_argument("--persona-test", type=Path, default=None,
                   help="Optional explicitly identified untouched Real Persona TEST JSONL")
    p.add_argument("--train-mixed", type=Path, default=Path(
        "data/dialogue/multitask_v107/multitask_train_v107.jsonl"))
    p.add_argument("--report", type=Path, default=Path(
        "results/frozen_sealed_test_v1012.json"))
    p.add_argument("--open-final-test", action="store_true",
                   help="Explicit opt-in to the single final TEST examination")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-unk-fraction", type=float, default=.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.gpu_dir / "model/tokenizer.json"
    sys.path.insert(0, str(args.gpu_dir.resolve()))
    result = evaluate(args)
    if not args.open_final_test:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
