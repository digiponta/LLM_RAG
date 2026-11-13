# LLM_RAG v0.9.4 — RoPE generation review

## Why
v0.9.3: frozen baseline 256-token normal attention model has Test answer NLL 2.59906; RoPE + same Legacy input has 2.49152 on 1,000 shared held-out QA rows (4.14% lower). Structured format yields essentially equal 2.49166 with RoPE. NLL does not establish conversation quality. v0.9.4 therefore compares actual generated dialogue with the same history and question for both frozen models.

## Windows PowerShell
    cd C:\Users\inoma\git\misc\LLM_RAG
    git fetch origin
    git switch feature/rope-generation-review-v0.9.4
    python -m unittest discover -s tests -v

Expected existing 173 tests + 12 newly added = 185 tests; requires user execution.

## Aggregate-only report (no dialogue text)
    python evaluate_rope_generation_v094.py `
      --baseline model/model-rag-legacy-paired-v0.9.2.pt `
      --rope-checkpoint model/model-rag-rope-legacy-v0.9.3.pt `
      --test-data data/dialogue/real_persona_structured_v092/paired_legacy_test_v092.jsonl `
      --sample 100 --output results/rope_generation_v094.json --device auto

## Local text-bearing review (private; do NOT commit)
    python evaluate_rope_generation_v094.py `
      --baseline model/model-rag-legacy-paired-v0.9.2.pt `
      --rope-checkpoint model/model-rag-rope-legacy-v0.9.3.pt `
      --test-data data/dialogue/real_persona_structured_v092/paired_legacy_test_v092.jsonl `
      --sample 20 --include-text `
      --output results/rope_generation_text_v094.json --device auto

Display the local report:
    $report = Get-Content results\rope_generation_text_v094.json -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($case in $report.cases) {
      $r = $case.review
      Write-Host '----------------------'
      Write-Host ('履歴: ' + ($r.history -join ' → '))
      Write-Host ('質問: ' + $r.question)
      Write-Host ('正解: ' + $r.reference_answer)
      Write-Host ('従来: ' + $r.legacy_response)
      Write-Host ('RoPE: ' + $r.rope_response)
      Write-Host ('NLL改善: ' + $r.nll_gain)
    }

## Analysis rules
- The two checkpoints receive the exact same original test row, history, question, reference, decoding hyperparameters and tokenizer. They only differ in learned Attention operations and resulting weights.
- The `summary.mean_example_nll_gain` is mean per-case answer-token NLL difference, unlike the corpus-token weighted Test NLL from evaluate_real_dialogue_v082. Do not directly compare their magnitudes.
- `response_change_rate`, `eos_rate`, `mean_chars`, `unique_response_ratio` and exact-string match do not measure semantic appropriateness.
- The text-bearing JSON has blank review rubric fields for each model (natural Japanese, question relevance, context reference, topic consistency; 0/1/2). Null means unreviewed. These are NOT auto-scored by the script.
- The evaluation is deterministic greedy with repetition penalty 1.3. Future runs could compare top-k sampling under controlled random seeds but are not conflated here.
- Source dialogues are private, context strings do not retain per-turn speaker IDs, and file-separated evaluation is not participant-independent.

## Next
Review blinded (A/B order randomized) generations for meaning, topic relevance and grammar. If improvement is small, a controlled model-capacity study is warranted; otherwise explore RoPE-optimized generation and stronger context supervision.
