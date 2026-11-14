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
