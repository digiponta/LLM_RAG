# LLM_RAG v0.10.9 — Task-Tagged Position Ratio Sweep and Premature EOS Diagnostics

## Prior experimental evidence

- v0.10.5, single-task Position SFT: 77.33% exact on 1,200 DEVELOPMENT original+counterfactual samples, 100% EOS termination.
- v0.10.7, six-task SFT (position 5%): good four-track Nagato/Persona/Chat/Evidence retention NLL but 0% Position exact; almost all prompts return an empty string.
- v0.10.8, same 5% mixture with explicit task markers: 419 unit tests PASS; four-task retention NLL passes, but still **0/1200** Position exact, 0.02 mean output characters, EOS termination 99.5%. Tags alone did not solve the problem.

These results do NOT prove that the model cannot learn multiple tasks. A plausible explanation is insufficient position-task exposure (200 of 4,000 train rows) or premature EOS; this branch isolates larger position fractions and measures first-step EOS explicitly.

## New experiment

Train three independent task-tagged v0.10.9 candidates from **the same frozen v0.10.0 checkpoint** and identical optimizer/seed/epochs/batch size as v0.10.8:

| Experiment | CHAT | KNOWLEDGE | PERSONA | EVIDENCE | POSITION | COPY |
|---|---:|---:|---:|---:|---:|---:|
| Reference v0.10.8 | 40% | 20% | 15% | 15% | 5% | 5% |
| v0.10.9 pos10 | 38% | 19% | 14% | 14% | **10%** | 5% |
| v0.10.9 pos20 | 33% | 17% | 13% | 12% | **20%** | 5% |
| v0.10.9 pos30 | 29% | 15% | 11% | 10% | **30%** | 5% |

The entire **fixed v0.10.8 400-row DEVELOPMENT** is reused, unchanged, for each variant's checkpoint selection. All variants are evaluated later on identical full 1,200-case POSITION DEVELOPMENT samples and the same 4-track Nagato/Persona/Chat/Evidence VALIDATION subsets. Training uses independent TRAIN-only source pools, stratifies position exposure equally between positions 1/5/10, caps repetition and checks source-origin separation. No new sources or final TEST opened; these are still development evaluations.

Increasing the position percentage necessarily decreases the other task percentages. This is a **ratio sweep**, not a causality proof of EOS shortcut bias or task-conditioning effects.

## 1. Checkout / tests

\`\`\`powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/position-ratio-sweep-v0.10.9
python -m unittest discover -s tests -v
\`\`\`

Expected **432** tests (419 prior + 13 new). GPU training, data generation and local verification have not been performed by the assistant.

## 2. First candidate: POSITION 10%

\`\`\`powershell
python prepare_position_ratio_v109.py \`
    --position-percent 10 \`
    --train-total 4000 --max-repeat 8 \`
    --validation data/dialogue/task_conditioned_v108/task_validation_v108.jsonl \`
    --output-dir data/dialogue/task_ratio_pos10_v109
\`\`\`

The report should show 400 POSITION train rows, stratified roughly 134/133/133 for character 1/5/10; fixed Validation count 400. All other task proportions follow the table.

\`\`\`powershell
python train_real_dialogue_v082.py \`
    --checkpoint model/model-rag-binding-v0.10.0.pt \`
    --source data/dialogue/task_ratio_pos10_v109/task_train_pos10_v109.jsonl \`
    --validation data/dialogue/task_ratio_pos10_v109/task_validation_fixed_v109.jsonl \`
    --min-answer-tokens 1 \`
    --max-train 0 --max-validation 0 \`
    --epochs 5 --batch-size 16 --learning-rate 0.00002 \`
    --rope --device auto \`
    --output model/model-rag-task-pos10-v0.10.9.pt \`
    --report results/task_train_pos10_v109.json
\`\`\`

Note: save happens only if validation NLL improves against epoch 0. The best v0.10.8 checkpoint was epoch 2; do not presume later epochs are better.

## 3. Position exact, first-step EOS, and counterfactual tracking

\`\`\`powershell
python evaluate_position_learning_v105.py \`
    --baseline model/model-rag-position-v0.10.5.pt \`
    --tuned model/model-rag-task-pos10-v0.10.9.pt \`
    --tuned-task-tag \`
    --dataset data/dialogue/copy_general_v103/copy_general_development_v103.jsonl \`
    --positions 1 5 10 --sample 0 \`
    --max-new-tokens 4 --device auto \`
    --output results/position_task_pos10_v109.json
\`\`\`

The v0.10.5 evaluator now also reports:
- \`mean_first_step_eos_probability\`: *before* generating the first answer token; high values directly explain near-empty generation.
- \`empty_response_fraction\`: actual empty free responses.
- Existing \`mean_gold_eos_probability\`: EOS given **gold target already supplied**, not to be confused with premature first-token EOS.
- Existing exact, first-token selection, EOS termination, natural/synthetic and changed-gold correct tracking.

## 4. Four-task retention and the original non-promoting gate

\`\`\`powershell
python evaluate_nagato_retention_v106.py \`
    --candidate-checkpoint model/model-rag-task-pos10-v0.10.9.pt \`
    --candidate-task-tag \`
    --max-rows 200 --generations 12 \`
    --batch-size 16 --max-new-tokens 96 \`
    --device auto \`
    --output results/nagato_retention_pos10_v109.json
\`\`\`

\`\`\`powershell
python evaluate_multitask_gate_v107.py \`
    --retention results/nagato_retention_pos10_v109.json \`
    --position results/position_task_pos10_v109.json \`
    --output results/position_gate_pos10_v109.json
\`\`\`

A gate pass is DEVELOPMENT-only and does not imply human-approved conversation quality or deployment readiness. The report key \`multitask_v107\` is reused for the candidate due to backward compatibility; inspect \`checkpoints\` for the actual v0.10.9 file.

## 5. Next candidates, only if needed

Run same steps with \`--position-percent 20\` and \`30\`, using distinct output folders/checkpoint names/validation reports:

- \`data/dialogue/task_ratio_pos20_v109\`, \`model/model-rag-task-pos20-v0.10.9.pt\`, \`results/position_task_pos20_v109.json\`
- \`data/dialogue/task_ratio_pos30_v109\`, \`model/model-rag-task-pos30-v0.10.9.pt\`, \`results/position_task_pos30_v109.json\`

Do not reuse filenames or overwrite the 10% artifacts. Keep baseline v0.10.8/v0.10.7 as experimental controls.

## Decision criteria

- **Position accuracy rises, first-step EOS drops, retention passes:** ratio exposure is a strong practical lever; continue with a more targeted intermediate percentage.
- **Position accuracy still 0% despite 20/30%:** diagnose training dynamics, task representation, loss imbalance and answer-start token distribution; consider a curriculum starting from position-trained v0.10.5 rather than raising weights further.
- **Position rises, retention fails:** report the Pareto tradeoff; consider staged rehearsal or retention regularization.
- **No checkpoint saved:** do not evaluate a nonexistent or stale path; inspect train report.

**Do not open the final Nagato TEST or treat source-quoted Nagato statements as externally verified scientific facts.**
