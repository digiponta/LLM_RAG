"""v0.10.7 retention acceptance gate for the experimental mixed checkpoint.

Uses identical-task NLL from the same evaluation report and position extraction
from a separately evaluated, source-disjoint DEVELOPMENT report. No training,
automatic promotion or source TEST access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = ("nagato_knowledge", "nagato_persona", "real_persona", "evidence_binding")


def decide(retention, position, *, max_nll_ratio=1.15, min_position_exact=.50,
           min_position_eos=.80, min_kind_exact=.35):
    if max_nll_ratio < 1 or not 0 <= min_position_exact <= 1 or not 0 <= min_position_eos <= 1 or not 0 <= min_kind_exact <= 1:
        raise ValueError("Invalid acceptance threshold")
    tasks = retention["tasks"]
    if not retention.get("validation_only"):
        raise ValueError("Retention report must be validation-only")
    results = {}
    for task in TASKS:
        item = tasks[task]
        if item["usable"] < 1:
            raise ValueError(f"No usable {task} examples")
        models = item["models"]
        old = models["binding_v100"]["answer_nll"]
        new = models["multitask_v107"]["answer_nll"]
        if old <= 0:
            raise ValueError("Invalid baseline NLL")
        ratio = new / old
        results[task] = {"baseline_nll": old, "candidate_nll": new,
                         "ratio": ratio, "pass": ratio <= max_nll_ratio,
                         "usable": item["usable"]}
    if position.get("version") != "v0.10.5":
        raise ValueError("Need explicit v0.10.5 position evaluation report")
    groups = position["groups"]["tuned"]
    pos_results = {}
    for key in ("all", "position_1", "position_5", "position_10",
                "natural", "synthetic"):
        item = groups[key]
        exact = item["exact_rate"]
        floor = min_position_exact if key == "all" or key.startswith("position_") else min_kind_exact
        pos_results[key] = {"cases": item["cases"], "exact_rate": exact,
                            "minimum": floor, "pass": item["cases"] >= 1 and exact >= floor}
    eos_rate = groups["all"]["eos_termination_fraction"]
    pass_eos = eos_rate >= min_position_eos
    ok = all(x["pass"] for x in results.values()) and all(
        x["pass"] for x in pos_results.values()) and pass_eos
    return {
        "version": "v0.10.7", "status": "DEVELOPMENT_GATE_PASS" if ok else "REJECT_REGRESSION",
        "accepted_for_deployment": False, "validation_only": True,
        "maximum_retention_nll_ratio": max_nll_ratio,
        "retention": results, "position": pos_results,
        "position_eos": {"rate": eos_rate, "min": min_position_eos,
                         "pass": pass_eos},
        "limitations": "Passing a DEVELOPMENT gate does not prove Nagato factual truth, "
                       "correct persona, dialogue quality, independent generalization or "
                       "copying accuracy. Passing never auto-promotes a checkpoint.",
    }


def run(args):
    inputs = (args.retention, args.position)
    if args.output.resolve() in {p.resolve() for p in inputs}:
        raise ValueError("Output path must differ from inputs")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError("Report already exists")
    retention = json.loads(args.retention.read_text(encoding="utf-8"))
    position = json.loads(args.position.read_text(encoding="utf-8"))
    result = decide(retention, position, max_nll_ratio=args.max_nll_ratio,
                    min_position_exact=args.min_position_exact,
                    min_position_eos=args.min_position_eos,
                    min_kind_exact=args.min_kind_exact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description="Non-promoting development acceptance gate")
    p.add_argument("--retention", type=Path, default=Path(
        "results/nagato_retention_multitask_v107.json"))
    p.add_argument("--position", type=Path, default=Path(
        "results/position_multitask_v107.json"))
    p.add_argument("--output", type=Path, default=Path(
        "results/multitask_retention_gate_v107.json"))
    p.add_argument("--max-nll-ratio", type=float, default=1.15)
    p.add_argument("--min-position-exact", type=float, default=.50)
    p.add_argument("--min-position-eos", type=float, default=.80)
    p.add_argument("--min-kind-exact", type=float, default=.35)
    p.add_argument("--overwrite", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
