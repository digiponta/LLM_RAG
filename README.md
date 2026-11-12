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
