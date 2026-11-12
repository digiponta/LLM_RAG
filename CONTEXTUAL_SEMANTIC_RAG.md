# LLM_RAG v0.5 — Contextual Semantic RAG

This release extends v0.4 with main-topic/focus-topic tracking and
context-sensitive retrieval, without evidence auditing, approval queues,
or verification workflows.

## What changes

- `宇宙についてどう考える？` establishes main topic = 宇宙.
- `時間については？` retains 宇宙 as the parent and sets focus = 時間.
- `もう少し詳しく` retrieves another unused candidate for the focus.
- Context-sensitive scoring favors texts mentioning the parent subject or
  related terms (時空・観測・粒子・重力) over incidental time mentions.
- If no context-compatible evidence is found, return a no-match response.
- Quotations are verbatim source material, not validated scientific facts.

**Scope:** The term semantic refers to a limited, manually defined vocabulary
of related words and ranking heuristics, not embedding-based semantic search.
An optional `--mode llm` remains a comparison mode using the previous
LLM_GPU generation path; it does not benefit from v0.5 contextual ranking.

## PowerShell

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/contextual-semantic-rag-v0.5
python -m unittest discover -s tests -v
python rag_persona_v05.py --docs data/knowledge --mode extractive
```

Test the following turns:

```text
宇宙についてどう考える？
時間については？
もう少し詳しく
あなたの名前は？
/status
/clear
もっと詳しく
/exit
```

Expected `/status` after the first three turns:
main_topic=宇宙 and focus_topic=時間.

Compare v0.4 and v0.5 outputs for irrelevant Windows boot-time quotations.
Nine new targeted tests are provided; the earlier branch contained 24 tests.
Full tests and actual conversation results on Windows have not yet been
verified in the development environment.

## Next: v0.6

Use independent evaluation cases and a better Japanese response generation
model for natural answers, while retaining an extractive fallback.
