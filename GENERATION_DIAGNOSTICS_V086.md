# LLM_RAG v0.8.6 — Generation Diagnostics

## Background

The user trained v0.8.4 on 197,582 real-dialogue pairs (64-token, character-based LLM_GPU). Held-out-by-file Test answer NLL fell from 3.506 to 2.722 and the mean matched-versus-shuffled-question NLL margin increased from 0.0103 to 0.1099. However, 20 generation samples still included generic responses and malformed Japanese. All 20 generated EOS under the particular decoding settings; this is not proof of semantic accuracy.

This experiment **does not retrain or modify** the model. It isolates the effect of the decoding strategy with the same test prompts and output token budget.

## Experiment

Five policies are compared on a fixed random sample of held-out Test prompts:

| Policy | Temperature | Top-k | Repetition penalty |
|---|---:|---|---:|
| greedy | 0.0 | none | 1.0 |
| penalty | 0.0 | none | 1.3 |
| sample | 0.8 | none | 1.0 |
| topk | 0.8 | 20 | 1.0 |
| topk_penalty | 0.8 | 20 | 1.3 |

For each policy, the script reports EOS rate, empty response rate, output diversity (unique response ratio and most common response share), length, repeated adjacent tokens and repeated bigrams. Sampling is reproducible through per-prompt random seeds; policies are not assumed to have identical distributions.

**No metric automatically determines semantic adequacy or grammaticality.** Those require inspection of the generated answers.

## Windows PowerShell

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/generation-diagnostics-v0.8.6

python -m unittest discover -s tests -v

# One model, short controlled diagnosis
python diagnose_generation_v086.py `
    --checkpoint model/model-rag-real-dialogue-full-v0.8.4.pt `
    --test-data data/dialogue/real_persona_v081/real_persona_test_v081.jsonl `
    --samples 30 `
    --max-new-tokens 48 `
    --device auto
```

Output: `results/generation_diagnostics_v086.json`. The report contains numeric statistics, hashed response identities and hashed case identifiers, but **no raw conversation text** by default.

To compare v0.8.2 with v0.8.4 under the same policies:

```powershell
python diagnose_generation_v086.py `
    --checkpoint model/model-rag-real-dialogue-full-v0.8.4.pt `
    --baseline model/model-rag-real-dialogue-v0.8.2.pt `
    --samples 30 `
    --output results/generation_comparison_v086.json `
    --device auto
```

For a locally inspectable report containing the original questions, reference answers, and generated responses, opt in explicitly:

```powershell
python diagnose_generation_v086.py `
    --checkpoint model/model-rag-real-dialogue-full-v0.8.4.pt `
    --samples 20 `
    --include-text `
    --output results/generation_diagnostics_text_v086.json `
    --device auto
```

**Privacy:** real dialogue text may include personal information. Keep text-bearing reports local, anonymize before sharing. No model/data is uploaded by this program.

## Interpretation and decision

- Compare the same 30 test prompts across policies; avoid choosing on raw NLL because checkpoints have unchanged weights and NLL does not depend on decoding.
- High diversity might reflect incoherent responses; low diversity indicates possible generic-answer collapse.
- High EOS rate can occur with answers that are too brief; read answer length together.
- If a decoding configuration significantly reduces generic answers **without degrading** human-inspected relevance and grammar, use it as the candidate for v0.8.7.
- If all policies produce unrelated/malformed answers, prioritize training objectives or model capacity rather than tuning decoding further.
- Source corpus split is dialogue-file-based, not independently verified by speaker/near duplicate.

New unit tests: 14; total expected after previous 97 = 111, pending local validation. Existing v0.8.4 checkpoint and stable extractive RAG are preserved.
