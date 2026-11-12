# LLM_RAG v0.6 — Grounded Response Planner and Dialogue Dataset Preparation

Extends the validated v0.5 contextual semantic retrieval implementation.
No audit, approval queue, or evidence verification workflow is introduced.

## Runtime changes

- `rag_persona_v06.py` retains the v0.5 main/focus topic routing.
- Uses a response plan with `EXCERPT`, `PERSONA`, `NO_MATCH`, or `LLM_COMPARISON`.
- On retrieval, selects a complete sentence from the returned original
  passage, preserving exact wording instead of asking the small LLM to
  synthesize ungrounded Japanese.
- Exposes the response strategy in `/status`.
- Default `extractive` works without CUDA. `--mode llm` still uses the
  older experimental LLM_GPU route and is expected to remain weak.

## Dialogue examples

`prepare_dialogue_v06.py` creates **synthetic quotation-style** QA pairs:
the question requests an original statement on a keyword and the answer
quotes the exact source post. Train and validation are split by post.
These pairs **are not verified scientific answers** and are not evidence of
genuine dialogue competence. No fine-tuning is performed by this script.

## PowerShell commands

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/response-planner-v0.6
python -m unittest discover -s tests -v
python rag_persona_v06.py --docs data/knowledge --mode extractive
```

Enter `宇宙についてどう考える？`, `時間については？`,
`もう少し詳しく`, `/status`, `/clear`, `/exit`.

To create the training-candidate files, run separately:

```powershell
python prepare_dialogue_v06.py --input data/knowledge/data-nagato.txt
```

This writes `data/dialogue/nagato_synthetic_train_v06.jsonl` and
`data/dialogue/nagato_synthetic_val_v06.jsonl`. Existing outputs are
protected unless `--overwrite` is explicitly passed.

## Current limitations

- The response planner is extractive and rule-based, not free-form generation.
- A selected quotation can be on-topic without directly answering a question.
- Vocabulary association and sentence choice are heuristics, not learned semantics.
- No new LLM_GPU training, GPU evaluation, or full-corpus quality study
  has yet been performed as part of v0.6.

## Next evaluation

Compare v0.5 and v0.6 on independent multi-turn prompts. Evaluate
topic continuity, relevance, textual source fidelity, answer usefulness,
and quality on topics not covered by the corpus.
