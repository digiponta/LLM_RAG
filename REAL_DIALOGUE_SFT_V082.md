# LLM_RAG v0.8.2 — Real Dialogue SFT (10k pilot)

## Scope

Train the existing LLM_GPU v0.3 model on short, locally prepared
Real Persona Chat pairs from v0.8.1; do not modify source dialogue, tokenizer,
v0.7 checkpoint, or the established extractive RAG runtime.

From the user's run: 13,583 parsed files, 197,595 training pairs,
23,533 validation pairs, and 23,233 held-out test pairs. These are pair
counts, not verified natural-language learning success.

## 1. Checkout and tests (Windows PowerShell)

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/real-dialogue-sft-v0.8.2
python -m unittest discover -s tests -v
```

Previous branch: 73 tests. This branch adds 12 tests (expected 85).

## 2. Pilot SFT

```powershell
python train_real_dialogue_v082.py `
    --gpu-dir ../LLM_GPU `
    --tokenizer ../LLM_GPU/model/tokenizer.json `
    --checkpoint model/model-rag-dialogue-v0.7.pt `
    --source data/dialogue/real_persona_v081/real_persona_train_v081.jsonl `
    --validation data/dialogue/real_persona_v081/real_persona_validation_v081.jsonl `
    --max-train 10000 `
    --max-validation 1000 `
    --epochs 3 `
    --batch-size 32 `
    --learning-rate 0.00005 `
    --device auto
```

Uses a deterministic random selection of up to 10,000 train and 1,000
validation rows. Each sample is encoded as `問:<utterance>\n答:<response>`;
the prefix is masked so only the answer and EOS contribute to CE loss.
Never truncate question or answer; reject full sequences beyond 64 tokens.
Default per-example UNK threshold: 10%, adjustable using
`--max-unk-fraction`. Check filter statistics before relying on results.

Output checkpoint: `model/model-rag-real-dialogue-v0.8.2.pt`
(saved only if validation NLL improves beyond the starting checkpoint).
Metrics: `results/real_dialogue_sft_v082.json`.

Existing outputs are protected unless `--overwrite` is explicitly passed.

## 3. Evaluate on the unused test split

Only run if a new checkpoint was saved:

```powershell
python evaluate_real_dialogue_v082.py `
    --gpu-dir ../LLM_GPU `
    --tokenizer ../LLM_GPU/model/tokenizer.json `
    --baseline model/model-rag-dialogue-v0.7.pt `
    --tuned model/model-rag-real-dialogue-v0.8.2.pt `
    --test-data data/dialogue/real_persona_v081/real_persona_test_v081.jsonl `
    --sample 500 `
    --generations 10 `
    --device auto
```

Output: `results/real_dialogue_eval_v082.json`.
Reports answer-token NLL, sample exact match, response length and EOS.
Raw conversations are **excluded** from the JSON report unless
`--include-text` is explicitly specified. When discussing results,
share only aggregate metrics unless conversation sharing is authorized.

## Design limits and next steps

- AA and AB represent dialogue speakers, not Nagato Yuki.
- This first pilot learns **one-turn responses**; v0.8.1 default
  `with_context=false` does not contain a prior turn.
- Conversation-file separation blocks direct source-file reuse but does
  not eliminate participant overlap, paraphrase/near-duplicate leakage,
  or shared topics.
- The model is a small character-token LM with context length 64.
  Real dialogues may have more unknown characters; inspect rejection
  rates and do not assume all 197,595 training pairs can be encoded.
- A lower NLL or exact-match rate does not prove correct, coherent
  or stylistically faithful answers. Review generations separately.
- Do not switch `rag_persona_v06.py` to this checkpoint until quality
  improves. Preserve the stable extractive response path.
- For larger experiments, set `--max-train 0` and `--max-validation 0`
  only after the pilot; use new output filenames to preserve comparisons.
