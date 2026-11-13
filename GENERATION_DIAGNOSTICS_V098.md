# LLM_RAG v0.9.8 — First-token and gold-prefix generation diagnostics

## Purpose
v0.9.7 definition QA improved concept Test NLL from 4.8858138304 to 4.3716549797 (8 held-out concept questions), but both models generated 0/8 meaningful concept explanations. Diagnose whether the failure is disproportionately at the **first answer token**, or persists even after the model is given the first N *correct* answer tokens.

## Scope and safeguards
- Baseline: v0.9.5 single-head RoPE Medium model (`model-rag-rope-medium-v0.9.5.pt`).
- Tuned: v0.9.7 definition QA fine-tuned model (`model-rag-concepts-v0.9.7.pt`).
- Use the same eight topic-disjoint (within the authored definition examples) concept questions.
- Freeze model weights. Do NOT train, mutate checkpoints, copy private data to GitHub or replace the baseline.
- Score (a) first-answer-token rank, greedy exact accuracy, probability and NLL at the prompt boundary, and (b) greedy-generated *remaining suffix* after providing the first 1, 4 or 8 correct answer tokens.
- For gold-prefix continuation, use only suffix tokens to calculate exact matches and leading-token match fraction. The gold-prefix experiment deliberately gives privileged reference information and is NOT a practical inference system.
- Tokenizer is the LLM_GPU character tokenizer, so 'first token' is usually the first answer character (it is not an answer phrase or semantic intention).
- Report suppresses all questions, reference text and generated text by default. `--include-text` enables raw text only in the user's local report. Keep that report private, particularly for Real Persona Chat.

## Step 1: Checkout and run unit tests (PowerShell)
    cd C:\Users\inoma\git\misc\LLM_RAG
    git fetch origin
    git switch feature/generation-diagnostics-v0.9.8
    python -m unittest discover -s tests -v

Expected 236 previous + 14 new = 250 tests. The actual Windows CUDA results require local execution.

## Step 2: Compare the same 8 held-out concept questions
    python evaluate_generation_diagnostics_v098.py `
      --baseline model/model-rag-rope-medium-v0.9.5.pt `
      --tuned model/model-rag-concepts-v0.9.7.pt `
      --test-data data/dialogue/concepts_v097/concept_test_v097.jsonl `
      --sample 8 --prefix-tokens 1 4 8 --max-new-tokens 80 `
      --output results/concept_generation_diagnostics_v098.json `
      --device auto

Console prints: first-token exact greedy accuracy, first-token mean target rank/NLL, and gold-prefix continuation exact fraction and mean matched leading reference-token coverage by prefix length.

## Step 3: Optional side-by-side local text review
    python evaluate_generation_diagnostics_v098.py `
      --baseline model/model-rag-rope-medium-v0.9.5.pt `
      --tuned model/model-rag-concepts-v0.9.7.pt `
      --test-data data/dialogue/concepts_v097/concept_test_v097.jsonl `
      --sample 8 --prefix-tokens 1 4 8 --max-new-tokens 80 `
      --include-text `
      --output results/concept_generation_diagnostics_text_v098.json `
      --device auto

To display the local report:
    $r = Get-Content results/concept_generation_diagnostics_text_v098.json -Raw -Encoding UTF8 | ConvertFrom-Json
    $r.cases | ForEach-Object {
      Write-Host "`nQuestion: $($_.question) | Reference: $($_.reference_answer)"
      Write-Host "Baseline first-token rank=$($_.baseline.first_token.target_rank), tuned rank=$($_.tuned.first_token.target_rank)"
      foreach ($n in @('1','4','8')) {
        if ($_.tuned.prefix.$n) {
          Write-Host "Gold-prefix=$n / expected suffix=$($_.tuned.prefix.$n.expected_suffix) / generated=$($_.tuned.prefix.$n.generated_suffix)"
        }
      }
    }

## Interpretation table

| Observation | Interpretation |
|---|---|
| First-token rank improves but gold-prefix continuation still fails | Opening-token selection is only part of the problem; downstream generation remains weak |
| First-token rank poor, but gold-prefix completion succeeds | Beginning-of-answer token choice is a likely bottleneck; investigate onset-focused training |
| Both first token and completion improve | Evidence of stronger conditional language modeling; still check free-generation relevance |
| Neither improves | Revisit tokenizer, training coverage, model capacity and task mismatch before mixed SFT |

## Scientific limitations
Eight examples are too few to establish statistical significance. An exact reference first character is not equivalent to the model giving a semantically suitable answer; many legitimate definitions start with different words. Prefix continuation is teacher-guided and can overstate what the free model can do. Evaluate full generated response relevance separately. No new claims about factual accuracy or model knowledge are made.
