# LLM_RAG v0.10.7 — Balanced Multi-Task SFT and Retention Gate

## Goal and experimental context

v0.10.6 measured degradation after pure copy (v0.10.3) and position (v0.10.5) SFT. On identical validation examples, v0.10.5 produced approximately 1-character responses even to long Nagato knowledge/persona and dialogue prompts, and suffered major NLL regression. Position extraction, however, reached 77.33% exact across 1,200 original/counterfactual development cases.

**This branch is an experiment**, not a claim of a working consolidated model. Start from preserved v0.10.0 and train one existing RoPE Medium with a deterministic mixture; reject candidates that substantially regress on held-out-by-source **DEVELOPMENT** data. Do not alter any old checkpoint or use final Nagato/Persona TEST materials.

## Data sources and mix ratios

| Task | Train source | Default target % |
|---|---|---:|
| Real Persona Chat | unified v0.9.9 TRAIN (only real_persona_chat rows, no mixed repeats) | 40 |
| Nagato Knowledge | Nagato v0.9.9 TRAIN source_attributed_qa | 20 |
| Nagato Persona | Nagato v0.9.9 TRAIN persona_style_from_source | 15 |
| Evidence Binding | v0.10.0 binding TRAIN evidence_direct | 15 |
| Position Extraction | v0.10.5 position TRAIN | 5 |
| Direct Copy | v0.10.3 generalized copy TRAIN | 5 |

Validation is sampled from the **corresponding existing VALIDATION/DEVELOPMENT** partitions and retains the same 40/20/15/15/5/5 ratio for model selection. This is a deliberate experimental compromise: global validation NLL depends on this mixture, while four-track evaluation protects against hiding a loss of any critical capability.

The mixer preserves original record IDs and conversation IDs as provenance fields, deterministically shuffles with an explicit seed, caps repeats via `--max-repeat`, and checks source/text separation. **Repeated records are NOT independent samples.** All generated text remains local and must not be committed. Original `data/knowledge/data-nagato.txt` is not reopened here; the existing v0.9.9 source-line partitions establish provenance. The final TEST split remains sealed.

## 1. Checkout and regression tests

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/balanced-multitask-v0.10.7
python -m unittest discover -s tests -v
```

Expected **404** tests (382 previously + 22 new); not executed by the assistant on Windows.

## 2. Prepare balanced TRAIN + DEVELOPMENT (initial target 4,000 / 400)

```powershell
python prepare_balanced_multitask_v107.py `
    --train-total 4000 `
    --validation-total 400 `
    --max-repeat 8 `
    --output-dir data/dialogue/multitask_v107
```

Expected quotas for train: 1,600 Real Persona / 800 Nagato Knowledge / 600 Nagato Persona / 600 Evidence Binding / 200 Position / 200 Copy. Development quotas: 160 / 80 / 60 / 60 / 20 / 20.

The generator may **refuse** these counts if a minority task has too few unique tokenizer-compatible examples for the eight-repeat cap. In that case examine the error and reduce `--train-total` (e.g. 2,000), or explicitly review the repetition cap. Do not blindly raise it.

## 3. Fine-tune from preserved v0.10.0 (not v0.10.5)

```powershell
python train_real_dialogue_v082.py `
    --checkpoint model/model-rag-binding-v0.10.0.pt `
    --source data/dialogue/multitask_v107/multitask_train_v107.jsonl `
    --validation data/dialogue/multitask_v107/multitask_validation_v107.jsonl `
    --min-answer-tokens 1 `
    --max-train 0 --max-validation 0 `
    --epochs 5 --batch-size 16 --learning-rate 0.00002 `
    --rope --device auto `
    --output model/model-rag-multitask-v0.10.7.pt `
    --report results/multitask_train_v107.json
```

The existing trainer uses answer-only masked cross-entropy **including EOS**, and writes the new model file only when *overall mixed DEVELOPMENT NLL* improves. Preserve all old v0.10.0/v0.10.3/v0.10.5 checkpoints. A mixed NLL improvement **is not sufficient** to promote the model.

## 4. Four-track retention evaluation against three frozen checkpoints

```powershell
python evaluate_nagato_retention_v106.py `
    --candidate-checkpoint model/model-rag-multitask-v0.10.7.pt `
    --max-rows 200 --generations 12 `
    --batch-size 16 --max-new-tokens 96 `
    --device auto `
    --output results/nagato_retention_multitask_v107.json
```

Scores are paired on identical validation samples for: Nagato knowledge, Nagato persona, Real Persona Chat and evidence binding. Compare **within each task**, not absolute NLL across tasks.

## 5. Position extraction, EOS and counterfactual tracking

```powershell
python evaluate_position_learning_v105.py `
    --baseline model/model-rag-position-v0.10.5.pt `
    --tuned model/model-rag-multitask-v0.10.7.pt `
    --dataset data/dialogue/copy_general_v103/copy_general_development_v103.jsonl `
    --positions 1 5 10 --sample 0 `
    --max-new-tokens 4 --device auto `
    --output results/position_multitask_v107.json
```

This scores 1,200 position-extraction cases including counterfactual swaps. Output includes actual one-character exact accuracy, position-specific accuracy and EOS termination, not merely changed output rate.

## 6. Development acceptance gate

```powershell
python evaluate_multitask_gate_v107.py `
    --retention results/nagato_retention_multitask_v107.json `
    --position results/position_multitask_v107.json `
    --max-nll-ratio 1.15 `
    --min-position-exact 0.50 `
    --min-position-eos 0.80 `
    --min-kind-exact 0.35 `
    --output results/multitask_retention_gate_v107.json
```

Default gate:
- Four-track per-task validation **answer NLL must be ≤115% of v0.10.0** (individually).
- Position extraction **overall and each of positions 1,5,10** ≥50% exact on DEVELOPMENT.
- Both natural and synthetic position extraction ≥35% exact.
- Position EOS termination ≥80%.

The gate reports `DEVELOPMENT_GATE_PASS` or `REJECT_REGRESSION`, but **never auto-promotes**; `accepted_for_deployment=false` even on a pass. Targets are *provisional research thresholds*, not statistically calibrated guarantees.

## 7. Monitor full copy ability separately

```powershell
python evaluate_generalized_copy_v103.py `
    --baseline model/model-rag-copy-general-v0.10.3.pt `
    --tuned model/model-rag-multitask-v0.10.7.pt `
    --dataset data/dialogue/copy_general_v103/copy_general_development_v103.jsonl `
    --sample 0 --device auto `
    --output results/copy_multitask_v107.json
```

This is **not part of the automatic gate** because v0.10.3's exact full copy rate was zero, making a relative non-regression gate meaningless. Report it separately before deciding to continue with copy curriculum.

## Interpretation / next stage

If the gate rejects, do **not** merge or use the mixed model as the default chat checkpoint. Review per-task NLL, sample length, EOS and distinct source counts. Sweep task ratios, lower learning rates, or use rehearsal/regularization against preserved v0.10.0; rerun on **development** only.

If it passes, manually review locally generated Nagato persona and dialogue samples and confirm literal knowledge attribution. Only once model settings and review criteria are fixed, run an **independent final TEST** evaluation. A data-source quote cannot be treated as verified scientific truth; copying, exact-match and EOS are **not** sufficient for factuality or natural dialogue.

