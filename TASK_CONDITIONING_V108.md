# LLM_RAG v0.10.8 — Task-Conditioned Multi-Task SFT

## Why this version exists

v0.10.7 balanced SFT significantly improved Nagato Knowledge validation NLL (3.609→2.720) and Persona NLL (4.155→3.538) relative to v0.10.0 and retained Real Persona Chat NLL (~2.257), but it learned **0% position extraction**, compared with 77.33% for v0.10.5 position-only. Most v0.10.7 position prompts generate EOS immediately and an empty answer. A high EOS termination rate is not success when no target character is produced.

v0.10.8 isolates **task identity as an input feature** while holding the v0.10.7 samples and ratios fixed. This is an explicit **controlled development experiment**, not a validated solution. Task tags are *ordinary textual strings*; the tokenizer, 2.1M parameter RoPE Medium architecture, and masked SFT loss are unchanged.

## Six task markers

| Source multitask | Task marker |
|---|---|
| real_persona | \`[TASK:CHAT]\` |
| nagato_knowledge | \`[TASK:KNOWLEDGE]\` |
| nagato_persona | \`[TASK:PERSONA]\` |
| evidence_binding | \`[TASK:EVIDENCE]\` |
| position | \`[TASK:POSITION]\` |
| copy | \`[TASK:COPY]\` |

The task marker appears before the row context; for empty-context records it becomes the context. If the row has a structured \`prompt_override\`, prepend the marker to the existing structured prompt instead. **Answers and EOS targets are never changed.** Task markers must be applied at inference to the candidate, not automatically to frozen untagged comparison models.

Both train/development use the *same 4,000/400 rows* as v0.10.7 and unchanged initial mixing ratio 40/20/15/15/5/5. So B (task-tagged) vs A (untagged) isolates the task-tag intervention, subject to new sampling/training randomness. The difference does not conclusively establish why 0% position happened; if 5% remains inadequate, a distinct ratio sweep is needed next.

## 1. Checkout and tests

\`\`\`powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/task-conditioned-v0.10.8
python -m unittest discover -s tests -v
\`\`\`

Expected **419 tests** (previous 404 plus 15 new). Local Windows and GPU execution remains pending.

## 2. Create the tagged versions of existing mixed data

Reuse v0.10.7 data exactly, which must exist:

\`\`\`powershell
python prepare_task_conditioned_v108.py \`
    --train data/dialogue/multitask_v107/multitask_train_v107.jsonl \`
    --validation data/dialogue/multitask_v107/multitask_validation_v107.jsonl \`
    --output-dir data/dialogue/task_conditioned_v108
\`\`\`

The generator checks all examples with the original tokenizer and context limit. It does **not** drop a task sample silently, which would change experimental ratios; on overflow or excessive unknown-token rate, an explicit error asks for inspection. Nagato and Persona TEST datasets stay sealed.

## 3. Train on exactly the tagged 4,000 / 400 examples

Start with the preserved v0.10.0 baseline (not v0.10.5 or previously tuned v0.10.7). Use the **same** learning rate, epochs and batch size as v0.10.7 so input tags are the main controlled variable.

\`\`\`powershell
python train_real_dialogue_v082.py \`
    --checkpoint model/model-rag-binding-v0.10.0.pt \`
    --source data/dialogue/task_conditioned_v108/task_train_v108.jsonl \`
    --validation data/dialogue/task_conditioned_v108/task_validation_v108.jsonl \`
    --min-answer-tokens 1 \`
    --max-train 0 --max-validation 0 \`
    --epochs 5 --batch-size 16 --learning-rate 0.00002 \`
    --rope --device auto \`
    --output model/model-rag-task-v0.10.8.pt \`
    --report results/task_train_v108.json
\`\`\`

As before, the trainer saves only a checkpoint with the best **overall mixed validation NLL** below the initial baseline. If no checkpoint is saved, do not proceed with evaluation using a nonexistent output.

## 4. Four-track retention evaluation — tag candidate only

\`\`\`powershell
python evaluate_nagato_retention_v106.py \`
    --candidate-checkpoint model/model-rag-task-v0.10.8.pt \`
    --candidate-task-tag \`
    --max-rows 200 --generations 12 \`
    --batch-size 16 --max-new-tokens 96 \`
    --device auto \`
    --output results/nagato_retention_task_v108.json
\`\`\`

Frozen v0.10.0/v0.10.3/v0.10.5 get their original untagged prompts; v0.10.8 candidate receives exactly its training marker (including teacher-forced NLL sequences). Thus model comparisons use *identical questions and target answers*, with explicitly different conditioning consistent with each model's training.

Internally the candidate retains the report key \`multitask_v107\` for backward compatibility with the existing retention gate. This is **a report field name**, not evidence the v0.10.7 model is being evaluated. The report \`checkpoints\` field gives the actual v0.10.8 checkpoint path and \`candidate_task_tag: true\`.

## 5. Position extraction — candidate tagged, frozen v0.10.5 untagged

\`\`\`powershell
python evaluate_position_learning_v105.py \`
    --baseline model/model-rag-position-v0.10.5.pt \`
    --tuned model/model-rag-task-v0.10.8.pt \`
    --tuned-task-tag \`
    --dataset data/dialogue/copy_general_v103/copy_general_development_v103.jsonl \`
    --positions 1 5 10 --sample 0 \`
    --max-new-tokens 4 --device auto \`
    --output results/position_task_v108.json
\`\`\`

Scores 1,200 original/swapped probes on the same v0.10.3 DEVELOPMENT source, comparing exact one-character target, EOS, natural vs synthetic and paired correct tracking.

## 6. Gate (same thresholds as v0.10.7)

\`\`\`powershell
python evaluate_multitask_gate_v107.py \`
    --retention results/nagato_retention_task_v108.json \`
    --position results/position_task_v108.json \`
    --output results/task_retention_gate_v108.json
\`\`\`

Criteria: every Nagato Knowledge, Persona, Real Persona and Evidence Binding task NLL ≤1.15× v0.10.0, position exact ≥50% overall and each 1/5/10, ≥35% in natural and synthetic, and EOS termination ≥80%. A gate pass is DEVELOPMENT ONLY, never deployment approval.

## Interpretation

- If Nagato and Persona NLL remains good and position exact increases materially vs v0.10.7, **task identification helps**, but still run further ratio sweeps and heldout prompt variants.
- If position remains near 0 with v0.10.8, task markers alone were insufficient. Next compare a higher position percentage (10%, 20%, 30%) **under the same task marker** and account for strong EOS shortcut bias.
- Inspect individual generated answers locally with \`--include-text\` and a distinct output filename for human/AI proxy review; do not commit private Real Persona Chat or generated original-text excerpts.
- Do not merge into the default chat checkpoint merely on NLL improvements or a Development Gate pass.
