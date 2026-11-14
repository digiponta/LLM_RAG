# LLM_RAG v0.10.4 — Evidence Position and Counterfactual Swap Probe

## Background

v0.10.3 learned 10-character output length and some leading characters, but free generation was 0/200 complete matches. A 30-example manual review revealed plausible Japanese continuations instead of literal source character copying; synthetic examples were especially unreliable.

This **diagnostic-only** experiment compares two existing, frozen checkpoints on three *new* question forms: "Which character is in position 1 / 5 / 10 of this provided source?" It also swaps evidence between held-out examples of the **same** kind while preserving question and position.

**Critical limitation:** Neither checkpoint has been trained to obey character-position questions. Failure is evidence that these frozen models do not obey these particular prompts, **NOT proof of an attention, RoPE, memory, or architecture defect.** The counterfactual experiment distinguishes whether output changes with the supplied evidence; change alone is not correctness.

## Setup

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/evidence-position-probe-v0.10.4
python -m unittest discover -s tests -v
```

Expected: 333 previous + 16 new = **349 tests**; local verification pending.

## Run the frozen-model position diagnostic

```powershell
python evaluate_evidence_position_v104.py `
    --baseline model/model-rag-binding-v0.10.0.pt `
    --tuned model/model-rag-copy-general-v0.10.3.pt `
    --dataset data/dialogue/copy_general_v103/copy_general_development_v103.jsonl `
    --positions 1 5 10 `
    --sample 30 `
    --max-new-tokens 8 `
    --device auto `
    --output results/evidence_position_probe_v104.json
```

For each selected 10-character DEVELOPMENT example, the tool constructs a question asking for exactly one Unicode character at the specified 1-based index. It scores both original evidence and a paired counterfactual using a different evidence string from the same natural/synthetic category. The original source remains intact.

Metrics:
- exact single-character answer fraction (strict comparison after decoding)
- first target token accuracy, rank and probability
- fraction of characters whose tokenizer encoding is exactly one token (to warn about character/token boundary mismatch)
- counterfactual output-change fraction between paired original/swapped evidence
- separate original/swapped and natural/synthetic breakdowns

For exact inspection locally, use `--include-text --output results/evidence_position_text_v104.json` instead. Without `--include-text`, report contains only hashes, metadata and aggregate metrics. Do not commit private data, output text or model checkpoints.

## Interpret

| Observation | Conclusion allowed |
|---|---|
| Zero accuracy at all three positions | Frozen models do not follow these new zero-shot instructions; architecture failure NOT established |
| Position 1 succeeds, 5/10 fail | Later character selection is harder under these prompts; needs a trained controlled position task |
| Output changes after swap but remains wrong | Model is input-sensitive but not accurate |
| Output changes and the new character is correct | Evidence-dependent position selection is demonstrated for these cases |
| Many characters use 2+ tokenizer tokens | Interpret token rank and character exact-match separately; consider character-level vs token-level experiment |

**Next, if zero-shot accuracy stays weak:** v0.10.5 should train an explicit controlled position-target dataset on source-disjoint TRAIN/VALIDATION before considering a pointer architecture. Once stable position selection is verified, consider a token-copy pointer gate using copied token scores and a separate coverage mechanism. Do not test against original Nagato TEST data during development.

The original Nagato TEST partition is not accessed by the script.
