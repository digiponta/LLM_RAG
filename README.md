# LLM_RAG v0.1 — Artificial Persona + Audit-Free RAG

A simple local retrieval-augmented artificial persona using **the existing
LLM_GPU Transformer weights**. No evidence auditing, approval queues, confidence
gate, human review, knowledge promotion or SKI dependency.

## Architecture

User -> character n-gram RAG (UTF-8 .txt/.md/.jsonl) -> persona/history prompt
-> LLM_GPU Tokenizer + LanguageModel.generate -> reply.

Search results supplement the prompt; **this is not model fine-tuning**.
The model may still hallucinate or ignore retrieved material.

## Setup (PowerShell)

Clone both repositories as siblings:

```powershell
cd C:Usersinomagitmisc
git clone https://github.com/digiponta/LLM_GPU.git
git clone https://github.com/digiponta/LLM_RAG.git
cd LLM_RAG
python -m pip install torch
```

Keep the checkpoint and tokenizer from the **same** training run at
`../LLM_GPU/model/model-gpu-v0.3.pt` and `../LLM_GPU/model/tokenizer.json`.
These binary artifacts are not shipped with this repository. If the newer
LLM_GPU model has a different API or tokenizer format, an adapter is needed.

Add UTF-8 files under `data/knowledge/`, optionally customize `persona.json`,
then run:

```powershell
python rag_persona.py --gpu-dir ../LLM_GPU --docs data/knowledge
python -m unittest discover -s tests -v
```

You can pass `--checkpoint ... --tokenizer ... --device cuda` to point to
your local model. Commands: `/reload` (rescan documents), `/clear`
(forget session turns), `/exit`.

## Important context-window limitation

The inspected LLM_GPU **main** branch implements a v0.3 **character-level**
tokenizer and a default **64-token** context. This is far too small for a
production RAG prompt and artificial persona. The model uses only the last
`context_length` tokens; the question is placed last to survive truncation.
For useful RAG fidelity, use a compatible, longer-context model and verify
answer quality with your corpus. The repository intentionally does not
pretend that the original 64-token model can process full retrieved chunks.

No audit workflow is present, but basic input parsing and transparent source
display remain. Retrieved documents are untrusted text; this small model
cannot reliably enforce the prompt's instruction boundaries.

## Nagato corpus

`data/knowledge/data-nagato.txt` is the original uploaded 2016
`@NagatFakeBot` corpus, retained verbatim (UTF-8, 430 lines). The default
`--docs data/knowledge` directory automatically includes it in retrieval.
The corpus contains authored opinions, fiction and hypotheses; retrieval
results are source text, not verified facts. Adding this file does **not**
train or update LLM_GPU weights. Run `/reload` after changing documents.

## Unified model training (v0.9.9)

Real Persona Chat and `data/knowledge/data-nagato.txt` now share the **same SFT data schema, tokenizer, single-head RoPE Medium model, answer-only loss and training command**. The old RAG route remains optional and separate from training. Prepare the local mixture with `python prepare_unified_sft_v099.py`; see [UNIFIED_NAGATO_PERSONA_SFT_V099.md](UNIFIED_NAGATO_PERSONA_SFT_V099.md) for the PowerShell commands for mixed SFT training and separate evaluation on Persona and Nagato held-out sets. Source attribution and original-line split grouping preserve the distinction between authored hypotheses/opinions and verified science. Generated local dialogues and model files must not be committed.

## v0.10.0 evidence-conditioned knowledge binding

Use `python prepare_knowledge_binding_v100.py` to prepare attributed Nagato direct and follow-up QA with verbatim `資料:` evidence using the same SFT schema as Real Persona Chat. Continue training from the v0.9.9 unified RoPE Medium checkpoint and evaluate **matched evidence-provided versus no-evidence held-out test sets**. See [KNOWLEDGE_BINDING_V100.md](KNOWLEDGE_BINDING_V100.md) for Windows commands and the distinction between oracle evidence conditioning and automatic retrieval. Training/validation data and checkpoints stay local.

## v0.10.1 Evidence Copy Diagnostics

Before further training, compare two **frozen** unified RoPE Medium checkpoints on faithful 10/30/50-character copying from Nagato TEST source lines. `evaluate_evidence_copy_v101.py` reports exact match, first-token, prefix, position, similarity and EOS measurements without exposing source text by default. It does not train, retrieve evidence, or claim knowledge; see [EVIDENCE_COPY_DIAGNOSTICS_V101.md](EVIDENCE_COPY_DIAGNOSTICS_V101.md).

## v0.10.2 Small Copy-Learning Probe

`prepare_copy_learning_v102.py` prepares 16 source-line-disjoint 10-character TRAIN and 8 DEVELOPMENT copy tasks (Nagato final TEST stays sealed). `train_copy_learning_v102.py` continues from the v0.10.0 checkpoint for a controlled micro-overfit test, tracking train/development NLL and actual greedy exact copy independently; it saves a diagnostic checkpoint even if development regresses. See [COPY_LEARNING_PROBE_V102.md](COPY_LEARNING_PROBE_V102.md) for full Windows instructions. No existing model or private data is overwritten.

## v0.10.3 Generalized Copy Learning

`prepare_generalized_copy_v103.py` builds 2,000 mixed natural/synthetic 10-character copy TRAIN examples and 200 source-separated DEVELOPMENT examples from original Nagato TRAIN and VALIDATION only. `train_real_dialogue_v082.py` continues the existing RoPE Medium SFT model; `evaluate_generalized_copy_v103.py` measures exact copying, first-token accuracy, prefix and positional character accuracy independently for natural and synthetic unseen strings. Final Nagato TEST remains untouched. See [GENERALIZED_COPY_V103.md](GENERALIZED_COPY_V103.md) for exact Windows commands and evaluation boundaries.

## v0.10.4 Evidence Position Probe

`evaluate_evidence_position_v104.py` compares frozen v0.10.0 and v0.10.3 checkpoints on explicit 1st/5th/10th Unicode-character extraction, with same-category counterfactual evidence swapping on the source-disjoint DEVELOPMENT set. It logs exact-character accuracy, token rank, and output sensitivity without exposing source text by default. These new instructions were *not trained*; failure is not a RoPE/attention diagnosis. See [EVIDENCE_POSITION_PROBE_V104.md](EVIDENCE_POSITION_PROBE_V104.md).

## v0.10.5 Position Learning SFT

`prepare_position_learning_v105.py` derives 1st/5th/10th-character question-answer pairs from source-disjoint v0.10.3 TRAIN and DEVELOPMENT data, excluding any source with unknown target characters. The shared trainer `train_real_dialogue_v082.py` now supports opt-in `--min-answer-tokens 1` (default remains 2) so one-character answers and EOS both receive SFT supervision. `evaluate_position_learning_v105.py` compares v0.10.3 and v0.10.5 on heldout development examples, measuring answer selection, exact one-character generation, gold-conditioned EOS and actual paired correct tracking after evidence swaps. This is a research experiment, not a general chat release. See [POSITION_LEARNING_V105.md](POSITION_LEARNING_V105.md).

## v0.10.6 Nagato Knowledge & Persona Retention

`evaluate_nagato_retention_v106.py` compares the **same three frozen checkpoints** (v0.10.0 binding, v0.10.3 copy, v0.10.5 position) on four **VALIDATION** tracks: Nagato attributed knowledge answers, Nagato persona-source style responses, Real Persona Chat dialogue, and oracle-evidence source binding. Reports answer-only token NLL and selected generation exact/length/EOS metrics, with local-only optional text review. It does not read the final Nagato TEST or train another checkpoint. Exact match and NLL are proxies rather than human-rated quality or factual truth. See [NAGATO_RETENTION_V106.md](NAGATO_RETENTION_V106.md).

## v0.10.7 Balanced Multi-Task SFT and Retention Gate

`prepare_balanced_multitask_v107.py` makes source-separated six-task TRAIN/VALIDATION mixes with initial ratios 40% Real Persona, 20% Nagato knowledge, 15% Nagato persona, 15% oracle evidence binding, 5% position, 5% copy, with deterministic sampling and a capped-repetition audit. Fine-tune from the preserved v0.10.0 checkpoint (never replace earlier models). `evaluate_nagato_retention_v106.py --candidate-checkpoint` now compares the v0.10.7 candidate on the same four tasks, and `evaluate_multitask_gate_v107.py` checks per-task NLL non-regression plus position extraction and EOS. Development pass never auto-promotes a checkpoint. See [BALANCED_MULTITASK_V107.md](BALANCED_MULTITASK_V107.md).

## v0.10.8 Task-Conditioned SFT

`task_conditioning_v108.py` adds ordinary text task markers (`[TASK:CHAT]`, `[TASK:KNOWLEDGE]`, `[TASK:PERSONA]`, `[TASK:EVIDENCE]`, `[TASK:POSITION]`, `[TASK:COPY]`) consistently to context or structured prompts without changing answers, tokenizer or RoPE Medium. `prepare_task_conditioned_v108.py` transforms exactly the same v0.10.7 4,000/400 mixed samples to isolate the task-identity intervention. Extend `evaluate_nagato_retention_v106.py` with `--candidate-task-tag` and `evaluate_position_learning_v105.py` with `--tuned-task-tag` so only the task-trained candidate receives matching input tags, leaving frozen baseline prompts untouched. Use existing development Retention Gate. See [TASK_CONDITIONING_V108.md](TASK_CONDITIONING_V108.md).

## v0.10.9 Position Ratio Sweep & First-Step EOS

v0.10.8 task markers alone preserved four-track Nagato/Persona/Chat/Evidence NLL but failed 1-character Position extraction (0/1200 exact, nearly empty responses). `prepare_position_ratio_v109.py` generates task-tagged six-way mixes from already-separated TRAIN sources with position ratios of **10%, 20% or 30%**, stratifying positions 1/5/10 and reusing the *exact same 400-row v0.10.8 DEVELOPMENT* for all candidates; original v0.10.0 and v0.10.8 remain frozen controls. The v0.10.5 position evaluator now reports EOS probability **at the first answer step** and actual empty-response fraction separately from gold-conditioned EOS. Continue to use 4-track retention and the non-promoting v0.10.7 gate. See [POSITION_RATIO_SWEEP_V109.md](POSITION_RATIO_SWEEP_V109.md).
