# LLM_RAG v0.3 — Extractive Persona

The Nagato fine-tuned LLM_GPU checkpoint still produces corrupted Japanese output. This version changes the **default response path** to extractive RAG. It keeps the original LLM generation path available for comparisons.

- `LineRAG` indexes individual posts instead of splicing unrelated lines into chunks.
- Standard greetings, name and reading-preference questions use small persona responses.
- Topical questions receive a **verbatim quotation** from a relevant corpus passage; source names are displayed. The quotation is an attributed opinion, not a verified fact.
- Unknown or unmatched queries are explicitly reported as such.
- No evidence audit, approval, review, training or knowledge-promotion pipeline.
- An optional `--mode llm` retains the existing trained GPU model and its known generation limitations.

## Windows PowerShell

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/extractive-persona-v0.3
python -m unittest discover -s tests -v
python rag_persona_v03.py --docs data/knowledge --mode extractive
```

Compare the trained LLM path:

```powershell
python rag_persona_v03.py --docs data/knowledge --mode llm \
  --gpu-dir ../LLM_GPU \
  --checkpoint model/model-rag-nagato-v0.2.pt \
  --tokenizer ../LLM_GPU/model/tokenizer.json --device cuda
```

The above optional multiline example uses shell-neutral backslashes for display; in Windows PowerShell use one line or backticks instead.

This is an interim baseline, **not conversational understanding**. A line can discuss a topic without answering the exact question. For robust dialogue the next step is question-answer dataset construction, independent holdout tests, and a larger-context instruction-trained model.
