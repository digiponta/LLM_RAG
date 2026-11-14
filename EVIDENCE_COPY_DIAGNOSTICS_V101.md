# LLM_RAG v0.10.1 — Evidence Copy Diagnostics

## Purpose

v0.10.0 improved evidence-provided held-out NLL from 3.46291 to 2.52513 (approximately 27.1%) but its 20 generated outputs all failed literal source fidelity. This step **does not train or modify models**. It isolates whether v0.9.9 and v0.10.0 frozen models can copy *exact Nagato source characters presented in the same prompt*.

The goal is to decide whether future training should prioritize elementary copying / source retention, passage selection, or answer synthesis. A good copying score does NOT establish retrieval, meaning or factuality; a poor score is evidence of weakness at a simpler prerequisite task.

## Update / unit tests

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/evidence-copy-diagnostics-v0.10.1
python -m unittest discover -s tests -v
```

Expected: 283 previous + 17 new = **300 tests** (not yet executed in the local Windows environment).

## Frozen-model benchmark

```powershell
python evaluate_evidence_copy_v101.py `
    --baseline model/model-rag-unified-nagato-v0.9.9.pt `
    --tuned model/model-rag-binding-v0.10.0.pt `
    --test-data data/dialogue/unified_v099/nagato_test_v099.jsonl `
    --widths 10 30 50 `
    --cases-per-width 12 `
    --max-new-tokens 64 `
    --repetition-penalty 1.0 `
    --output results/evidence_copy_diagnostics_v101.json `
    --device auto
```

Uses the v0.9.9 **Nagato TEST partition** only, not training source lines. For each original source quotation, selects an exact prefix of length 10, 30, or 50 Unicode characters, then constructs the same `前:資料:...` and `問:...` and `答:` prompt schema used by Real Persona SFT. Each selected **original source line** is used once across all lengths to avoid case duplication. Selects 50-character cases first, then 30, then 10. If there are too few eligible source lines, output contains smaller per-length case counts; never fabricate data to fill quotas. Max generation length is at least expected token count + 8, and all source/output examples must satisfy the 256-token context and low-UNK checks.

The default report includes only hashed IDs, counts, accuracy metrics and non-text metadata. It does not expose Nagato or private dialogue text by default. Optional local `--include-text` adds full prompt, expected text, and generated text: review it locally, never commit it.

## Metrics

For **each length** and each model:

- `exact_rate`: fraction of generated strings identical to the provided source excerpt
- `first_token_accuracy` and `mean_first_token_rank`: whether the first output token matches that exact excerpt
- `mean_prefix_coverage`: fraction of expected initial characters reproduced before first mismatch
- `mean_position_accuracy`: positional character matches divided by expected characters
- `mean_sequence_similarity`: rough text similarity (SequenceMatcher), diagnostic only
- `mean_generated_chars` and `eos_fraction`: answer length and EOS behavior

Both models receive the **same examples** and deterministic greedy decoding, with repetition penalty 1.0 to avoid interfering with literal repetition. Existing checkpoint files are read-only. There is no automatic threshold that claims success.

To view only the aggregate local summary:

```powershell
$r = Get-Content results/evidence_copy_diagnostics_v101.json -Raw -Encoding UTF8 | ConvertFrom-Json
$r.evaluation | ConvertTo-Json -Depth 8
```

To inspect local generated examples without changing model weights, use a distinct output path and `--include-text`.

## Interpretation and boundaries

- If 10-character exact copy is weak, investigate the output loss and token-level generation mechanism before attempting complex question answering.
- If 10 works but 30/50 fail, study long-sequence source retention and output termination.
- If all lengths copy well but knowledge answers remain poor, prioritize evidence selection and response construction.
- The original Nagato corpus includes fiction/hypotheses. Verbatim copying is **not** a truth test.
- This task uses oracle prompt content. It does not show the model can retrieve, infer, or internally recall the answer.
- The test split is source-line held out from the prior Nagato training transformation; it can still overlap thematically.
