# v0.10.13 — data-nagato.txt Proper-Name Direct Quote

When a user question literally mentions a recognized proper name and that name appears in **data/knowledge/data-nagato.txt**, \`chat.py\` **bypasses LLM generation** and returns exactly **one original source sentence** containing that name. This branch implements "１文" (one sentence), not an arbitrary 200-character RAG chunk. The sentence is attributed to the local, **unverified** source and is not asserted to be scientific truth.

## PowerShell

\`\`\`powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/nagato-entity-direct-v0.10.13
python -m unittest discover -s tests -v
python chat.py --device auto --verbose
\`\`\`

Expected **474 tests** (458 previous plus 16 new); execution pending the user's Windows results.

Default direct-hit mode does **not** require \`--rag\`. With \`--rag\`, direct-hit priority still takes precedence over RAG passages and model generation. A matching case works like:

\`\`\`text
You> イスカンダルとは？
AI> （data-nagato.txt内のイスカンダルを含む原文の１文）
[Nagato direct quote: entity=イスカンダル, source=data-nagato.txt:..., source statement, not independently verified; LLM bypassed]
\`\`\`

The source sentence actually returned depends on local corpus content. This illustrative placeholder is **not** a generated output.

\`\`\`text
You> こんにちは
AI> （従来のLLMによる応答）
\`\`\`

When no recognized proper name appears in the question or source, the legacy model generation/RAG path is unchanged. The \`/position\` and \`/copy\` experiments bypass the new shortcut.

## Where names come from

Automatic extraction is **heuristic**, not full Japanese morphological Named Entity Recognition:
- Known seed name list includes 長門有希, 攻殻機動隊, イスカンダル, コスモクリーナー, and other distinctive forms (only if they truly occur in original source sentence).
- Distinctive long katakana and capitalized Latin names found in the file, and quoted names/titles, are candidate entities.
- Common katakana generic words such as パソコン, ストレス, データ, システム are blocked from heuristics to reduce spurious matches.
- To recognize additional Japanese kanji proper nouns, add a local file \`data/knowledge/nagato_entities.txt\` with one name per line and launch with \`--nagato-entities data/knowledge/nagato_entities.txt\`. The file is **optional**, and names still must occur in the source text. Local file contents are never sent to GitHub or an external service.

For example, local entity lexicon:

\`\`\`text
# 固有名詞を１行につき１件
長門有希
攻殻機動隊
\`\`\`

## Additional CLI options

\`\`\`powershell
python chat.py --nagato-source data/knowledge/data-nagato.txt
python chat.py --nagato-entities data/knowledge/nagato_entities.txt
python chat.py --no-nagato-direct
python chat.py --rag --docs data/knowledge
\`\`\`

\`/reload\` refreshes the direct Nagato index AND optional RAG index; \`/status\` reports direct mode. \`/sources\` lists \`data-nagato.txt\` after direct hits. \`--no-nagato-direct\` restores ordinary RAG/LLM-only behavior.

## Interpretation / limitations

- A matching entity chooses **the first original sentence containing the longest matched entity in source order**. If there are multiple assertions or meanings, this is **not** best semantic relevance ranking.
- Only one original sentence is returned; no summary, no automatic truth verification, no LLM answer blending. False matches and missed kanji-only names are possible without a validated Japanese NER resource.
- This is a deterministic **retrieval shortcut**, not evidence that v0.10.11 learned the answer or improved free-generation quality.
- Corpus text may contain unverified hypotheses, invented science, opinions, and fictional characters; output must be attributed to its source.
- The frozen v0.10.11 checkpoint remains unchanged; no retraining and no new independent model scoring.
