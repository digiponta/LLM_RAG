# LLM_RAG v0.4 — Context-Aware Artificial Persona

Non-audited extractive dialogue with intent classification, topic memory, and
novelty-aware retrieval.

## Features

- Distinguish greeting, identity, preference, opinion, definition, topic and follow-up.
- Preserve the preceding topic when asked `もう少し詳しく`.
- Explicitly change topic for `時間については？`.
- Rank original Nagato posts using literal topic match, character n-grams,
  query-intent hints, and avoidance of previously quoted passages.
- Attribute retrieved passages as original writings rather than validated facts.
- `/status`, `/clear`, `/reload`, `/exit`.
- Optional `--mode llm` retains the original LLM_GPU path for comparison,
  but its previously observed generation quality is poor.

No verification/approval/audit workflow, and no new parameter training.
This is **rule-based contextual retrieval**, not full natural-language understanding.
If a post mentions a topic without answering the question, the extractive mode
will still quote it rather than fabricate an explanation.

## Windows PowerShell

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/context-persona-v0.4
python -m unittest discover -s tests -v
python rag_persona_v04.py --docs data/knowledge --mode extractive
```

Suggested evaluation conversation:

```text
宇宙についてどう考える？
もう少し詳しく
時間については？
もう少し詳しく
あなたの名前は？
/status
/clear
もっと詳しく
/exit
```

LLM comparison, if desired:

```powershell
python rag_persona_v04.py --docs data/knowledge --mode llm --gpu-dir ../LLM_GPU --checkpoint model/model-rag-nagato-v0.2.pt --tokenizer ../LLM_GPU/model/tokenizer.json --device cuda
```

LLM mode does **not** yet integrate contextual ranking into the prompt.
The two modes are intentionally kept distinct for evaluation.
