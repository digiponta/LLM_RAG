# LLM_RAG v0.9.1 — History-visible generation review

## Goal
Investigate why v0.9.0 context-256 training reduces answer NLL (3.016 → 2.599 on matched test prompts) while generated responses remain often generic, ungrammatical or unrelated.

Read the real preceding turns alongside the question, expected answer, and generated answers with and without history. Keep checkpoint weights frozen. This is a diagnostic, NOT speaker/turn embedding training.

## PowerShell commands
    cd C:\Users\inoma\git\misc\LLM_RAG
    git fetch origin
    git switch feature/context-generation-review-v0.9.1
    python -m unittest discover -s tests -v

Generate an aggregate-only report (no private utterances):
    python evaluate_context_generation_v091.py --checkpoint model/model-rag-context256-history-v0.9.0.pt --sample 30 --output results/context_generation_v091.json --device auto

Generate a LOCAL text-bearing report:
    python evaluate_context_generation_v091.py --checkpoint model/model-rag-context256-history-v0.9.0.pt --sample 20 --include-text --output results/context_generation_text_v091.json --device auto

Inspect locally with PowerShell:
    $report = Get-Content results\context_generation_text_v091.json -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($case in $report.cases) {
        $r = $case.review
        Write-Host '--------------------------'
        Write-Host ('履歴: ' + ($r.history -join ' → '))
        Write-Host ('質問: ' + $r.question)
        Write-Host ('正解: ' + $r.reference_answer)
        Write-Host ('履歴なし: ' + $r.without_context)
        Write-Host ('履歴あり: ' + $r.with_context)
        Write-Host ('NLL差 (plain - context): ' + $r.context_nll_advantage)
    }

## Interpretation
- Mean context NLL advantage is NLL(answer | question alone) minus NLL(answer | true history, question), calculated on the same reference response for the same model.
- Response-change rate indicates whether output changes, not whether it is relevant.
- Use generated examples to distinguish missing reference resolution, generic response, grammatical drift, and semantic topic switching.
- The original context joined utterances using '／' and does NOT keep per-utterance speaker IDs. Do not infer a specific speaker label for each history segment. question_speaker and answer_speaker are still available.
- Existing checkpoint has no explicit positional embeddings. This experiment does not change that and cannot by itself prove positional encoding is needed.
- Test split is dialogue-file-based and might overlap by speaker/near duplicate. Text-bearing reports may include sensitive real conversations; keep local and redact before sharing.

## Next engineering decision
Based on the review, decide whether to prioritize speaker/turn-aware dataset regeneration, position encoding, or model capacity scaling. Introduce new model structures as separate ablations rather than conflating them with new training data.
