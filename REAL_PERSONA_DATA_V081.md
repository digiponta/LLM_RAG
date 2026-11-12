# LLM_RAG v0.8.1 — Import Real Persona Chat (Local Only)

## Evidence from the user's PowerShell listing

- 8,971 JSON files are listed (about 70.9 MB of file-size totals).
- Representative structure: `{dialogue_id, interlocutors, utterances}`.
- Each utterance has `utterance_id`, `interlocutor_id`, `text`, `timestamp`.
- Two speaker codes `AA` and `AB` appear in the sample.
- The listing alone does not establish the total turn count or usable examples.

## Important model boundaries

LLM_GPU uses a 64-token character tokenizer. v0.8.1's converter
therefore selects short, adjacent cross-speaker pairs fitting a
64-character token budget. Original source files are NOT modified or uploaded.
Speaker roles AA/AB are not assumed to represent the Nagato persona.

Conversion is **not training**. Do not pass this output directly to the
old `train_dialogue_v07.py`, which expects synthesized Nagato topic/answer
rows. A separate real-dialogue SFT adapter is required next.

## PowerShell

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/real-persona-dialogues-v0.8.1
python -m unittest discover -s tests -v
python prepare_real_persona_v081.py `
    --input-dir ../LLM_CHA/data-src/real-persona-chat/real_persona_chat/dialogues `
    --output-dir data/dialogue/real_persona_v081 `
    --max-context 64
```

Outputs:

- `data/dialogue/real_persona_v081/real_persona_train_v081.jsonl`
- `data/dialogue/real_persona_v081/real_persona_validation_v081.jsonl`
- `data/dialogue/real_persona_v081/real_persona_test_v081.jsonl`
- `data/dialogue/real_persona_v081/real_persona_report_v081.json`

Use `--with-context` to include the prior utterance when it fits;
otherwise leave it off while experimenting with 64-character contexts.
Output files are protected against overwrite by default.

## Extraction and split rules

- Split by **dialogue file**, not by individual utterance.
- Deterministic hash split: approximately 80% train, 10% validation, 10% test.
- Use adjacent turns from different speakers; preserve their two role IDs.
- Retain distinct short pairs, drop exact duplicate question/answer/context
  triples across all outputs, and skip badly formatted files or replacement
  characters (`�`).
- Guard against prompt+answer overflow for a character-level tokenizer.
- Report counts and filtering statistics without echoing private conversation
  content into console summaries.

## Caveats

- Some Real Persona Chat files might contain names or personal information;
  handle output files locally and respect the underlying corpus license.
- The script does not de-identify or content-moderate dialogue content.
- Original dialogues are not necessarily in the style of Nagato Yuki.
- Even a dialogue-file split is not guaranteed to be independent where
  speakers, topics, or near-duplicate conversations span files.
- A 64-token window cannot encode much multi-turn context. Future model
  scaling and targeted dialogue fine-tuning are separate work.

## Next checks

Inspect the report's number of parsed files, candidate pairs, usable pairs,
duplicate counts and train/validation/test sizes before training.
