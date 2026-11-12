# LLM_RAG v0.8.3 — Generic Answer Bias and Question Conditioning

Observed on 10 held-out conversations after v0.8.2 real dialogue SFT:
EOS 10/10 with repetition penalty 1.3, while 'そうなんですか？' appeared
3 times and many other answers did not address their inputs.
The 499-case answer-token NLL was 5.37194 (v0.7) versus 3.49926
(v0.8.2). This does not establish useful conversational responses.

## What is new

- `rebalance_responses_v083.py`: analyze normalized answer frequency in
  the **training** JSONL only, then create a deterministic training
  subset capped by count of identical normalized answers. The report
  omits private answer strings by default (uses truncated SHA-256).
- `evaluate_conditioning_v083.py`: use previously held-out **test**
  JSONL, compare answer-only NLL when the original response is paired
  with its true question vs a different question from another row.
- No changes to the stable RAG runtime, tokenizer or checkpoints.
- 12 new unit tests.

## Windows PowerShell

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/response-conditioning-v0.8.3
python -m unittest discover -s tests -v

# 1. Baseline question-conditioning without further training
python evaluate_conditioning_v083.py `
  --baseline model/model-rag-real-dialogue-v0.8.2.pt `
  --test-data data/dialogue/real_persona_v081/real_persona_test_v081.jsonl `
  --sample 100 --device auto

# 2. Analyze frequency and create 10k training-only subset
python rebalance_responses_v083.py `
  --train data/dialogue/real_persona_v081/real_persona_train_v081.jsonl `
  --sample 10000 --max-per-answer 3

# 3. Train using SAME trainer but an alternative output
python train_real_dialogue_v082.py `
  --source data/dialogue/real_persona_v083/train_balanced_10000.jsonl `
  --validation data/dialogue/real_persona_v081/real_persona_validation_v081.jsonl `
  --checkpoint model/model-rag-dialogue-v0.7.pt `
  --output model/model-rag-balanced-v0.8.3.pt `
  --report results/real_dialogue_balanced_v083.json `
  --max-train 0 --max-validation 1000 `
  --epochs 3 --batch-size 32 --device auto

# 4. Evaluate response-conditioning of both checkpoints if saved
python evaluate_conditioning_v083.py `
  --baseline model/model-rag-real-dialogue-v0.8.2.pt `
  --candidate model/model-rag-balanced-v0.8.3.pt `
  --test-data data/dialogue/real_persona_v081/real_persona_test_v081.jsonl `
  --sample 100 --output results/conditioning_balanced_v083.json `
  --device auto
```

Expected suite: 85 previous tests + 12 new = 97, pending local execution.
All output names are separate from v0.8.2 and default to no overwrite.

## Interpretation

- Mean NLL margin = NLL(actual answer | swapped question) minus
  NLL(actual answer | matched question). Larger positive margin
  indicates stronger question conditioning; near-zero indicates weaker
  sensitivity. This is a diagnostic proxy, not a measure of truth.
- Frequency balancing reduces repeated *exact normalized strings*;
  it cannot guarantee fewer semantically equivalent generic replies.
- Balanced 10k is a different training distribution from the original
  10k pilot: compare on the SAME untouched evaluation split. Balanced
  selection might return fewer than 10k when frequency caps bind.
- Mismatched prompts may have different lengths and content. The
  diagnostic controls the ANSWER but not all contextual confounders.
- The test split was constructed by source dialogue file, not speaker.
  Report results as held-out-by-file, not fully independent.
- Keep raw chat JSONL local. The report defaults to hashes rather
  than source text. Use `--expose-text` only when local inspection requires it.

## Next step

If balanced data improves conditional margin and subjective response
relevance, test a larger 20k/50k controlled comparison. Otherwise
investigate model capacity and longer context. Never turn on the
generative model in the stable RAG by default without inspecting
actual generated conversations.
