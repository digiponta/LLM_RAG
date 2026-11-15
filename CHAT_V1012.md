# LLM_RAG chat.py — Frozen v0.10.11 Unified CLI (v0.10.12)

The **default model** for \`python chat.py\` is now the frozen local RoPE model \`model/model-rag-position-rehearsal-v0.10.11.pt\`. No checkpoint is trained, modified or promoted. Older checkpoints can still be selected using \`--checkpoint\`. Legacy RoPE/MHA/intent load dispatch, optional local RAG, status, reset, source listing and sliding history remain available.

## Start on Windows PowerShell

\`\`\`powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/chat-v0.10.12
python -m unittest discover -s tests -v
python chat.py --device auto --verbose
\`\`\`

Expected **458 tests** (443 previous + 15 new) pending local verification.

## CLI commands

- \`/mode chat\`: ordinary conversational generation (default)
- \`/mode knowledge\`: experimental *prompt wording*, not guaranteed verified source knowledge
- \`/mode persona\`: experimental *prompt wording*, not a certified role/persona evaluation
- \`/mode evidence\`: experimental source-only wording; without \`--rag\` no evidence is retrieved, and response is flagged unverified
- \`/position 5 ABCDEFGHIJ\`: original training-format single-character extraction request; answers via the model only, **not Python substring slicing**. Only positions 1, 5, 10 and evidence strings exactly 10 characters long match trained data. \`temperature=0\`, \`repetition_penalty=1\`, \`max_new_tokens<=4\` match benchmark settings. Expected answer for example is E but model predictions may be wrong. Position commands **do not contaminate chat history**.
- \`/copy ABCDEFGHIJ\`: use original v0.10.3 exact-copy prompt (10-character text), experimental **not reliable**; v0.10.11 Copy DEVELOPMENT exact was 0/200. This uses model generation, not a Python copy bypass. Does not contaminate chat history.
- \`/reset\`: clear two-turn context and source list.
- \`/status\`: display checkpoint, history count, mode, RAG state.
- \`/sources\`: display currently retrieved local source file identifiers.
- \`/reload\`: refresh local RAG passages if \`--rag\` enabled.
- \`/exit\`: quit.

\`/mode\` changing the answer prompting mode clears history. The default plain chat is **not task-tagged** because v0.10.11's two-stage Position-First + rehearsal training used untagged v0.10.7 mixes.

## Optional RAG

\`\`\`powershell
python chat.py --rag --docs data/knowledge --top-k-docs 1 --device auto
\`\`\`

This is best-effort local retrieval, **not** evidence binding guaranteed by the unverified source QA corpus; retrieved documents are unverified and not a guarantee of factual truth. The old \`/sources\` command still exposes sources.

## Frozen model status

As of the prior user-run experiments:
- 443 previous unit tests PASS.
- Development Position exact **704/1,200 = 58.67%** (positions 1/5/10: 61.5/59.5/55%).
- Development four-track Retention Gate PASS for Nagato Knowledge, Nagato Persona, Real Persona Chat, Evidence Binding.
- A **one-time** Nagato original-line held-out TEST recorded improved answer-token NLL on Nagato Knowledge, Persona and oracle Evidence Binding vs v0.10.0.
- **Copy 0/200 exact** in Development despite improved first-token accuracy.
- Real Persona independent TEST, new Position/Copy TEST and natural multi-turn quality have **not** been established.

This is a CLI convenience release, not a new model checkpoint or evidence of full general chat competence.
