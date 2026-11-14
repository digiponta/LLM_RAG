# LLM_RAG v0.10.6 — Nagato Knowledge & Persona Retention

## Goal

Recenter the project on the original `data/knowledge/data-nagato.txt` goals: one LLM should respond to attributed Nagato knowledge questions and persona-style prompts while preserving ordinary Real Persona Chat ability. Copy and position-extraction improvements (v0.10.3, v0.10.5) should not be mistaken for retained knowledge or persona. This stage is **evaluation only**: no weights are changed.

## Data lineage

`data/knowledge/data-nagato.txt` → `prepare_unified_sft_v099.py` → `data/dialogue/unified_v099/nagato_{train,validation,test}_v099.jsonl` (split by original source line). It creates `source_attributed_qa` and `persona_style_from_source`. Real Persona Chat validation comes from `unified_validation_v099.jsonl`. `prepare_knowledge_binding_v100.py` creates explicit oracle-evidence questions in `binding_validation_v100.jsonl`.

v0.10.3 uses the Nagato TRAIN/VALIDATION rows to generate copy tasks; v0.10.5 uses their 10-character outputs to generate position targets. The `data-nagato.txt` source is therefore still in the lineage, but *copy success does not establish recall of Nagato knowledge*. Corpus statements are **unverified attributions**, not verified external facts.

## Compare the same models on four tasks

| Track | Local Validation JSONL | Row selection | Interpretation |
|---|---|---|---|
| Nagato Knowledge | `unified_v099/nagato_validation_v099.jsonl` | `source_attributed_qa` | Recall an attributed source response *without oracle evidence* |
| Nagato Persona | same Nagato validation | `persona_style_from_source` | Reproduce source-based style response; a lexical proxy, **not** human persona judgement |
| Real Persona | `unified_v099/unified_validation_v099.jsonl` | `source_kind=real_persona_chat` | Response continuation on dialogue validation; not a human-rated conversational metric |
| Evidence Binding | `binding_v100/binding_validation_v100.jsonl` | `evidence_direct` | Respond when the correct source quote is supplied as context; **not retrieval** |

Models compared: `model-rag-binding-v0.10.0.pt`, `model-rag-copy-general-v0.10.3.pt`, `model-rag-position-v0.10.5.pt`, all using the exact same tokenizer and single-head RoPE Medium checkpoint loader.

## 1. Checkout and regression tests

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/nagato-retention-eval-v0.10.6
python -m unittest discover -s tests -v
```

Expected **382 tests** (366 preceding + 16 new), pending Windows local verification.

## 2. Run retention evaluation

```powershell
python evaluate_nagato_retention_v106.py `
    --nagato-validation data/dialogue/unified_v099/nagato_validation_v099.jsonl `
    --unified-validation data/dialogue/unified_v099/unified_validation_v099.jsonl `
    --binding-validation data/dialogue/binding_v100/binding_validation_v100.jsonl `
    --binding-checkpoint model/model-rag-binding-v0.10.0.pt `
    --copy-checkpoint model/model-rag-copy-general-v0.10.3.pt `
    --position-checkpoint model/model-rag-position-v0.10.5.pt `
    --max-rows 200 --generations 12 --batch-size 16 `
    --max-new-tokens 96 --device auto `
    --output results/nagato_retention_v106.json
```

This reads **only the existing VALIDATION files**, rejects filenames containing `test`, `holdout` or `final`, and never opens the Nagato original final Test split. It selects a seeded identical subset of at most 200 examples per track and computes exact **answer-token teacher-forced NLL** for each frozen model; rejected examples are counted. Generation samples are the same for all three models, using greedy decoding and repetition penalty 1.0.

Do **not** compare absolute NLL across *different tasks*. Compare models **within each identical task**. A fall in NLL is not automatically better free-generation quality.

## 3. View concise scorecard

```powershell
$r = Get-Content results/nagato_retention_v106.json -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($t in 'nagato_knowledge','nagato_persona','real_persona','evidence_binding') {
    Write-Host "`n=== $t ==="
    $r.tasks.$t.models | ConvertTo-Json -Depth 5
}
```

## 4. Optional manual sample review

To see reference answers and generated responses, rerun with `--include-text` **and a distinct output path**, for example:

```powershell
python evaluate_nagato_retention_v106.py `
    --generations 12 --max-rows 200 --device auto `
    --include-text `
    --output results/nagato_retention_review_text_v106.json
```

Original data can include personal dialogue and unverified source statements; keep this report **local and private**. For persona, human or proxy-assisted evaluation should explicitly mark subjective judgments as such and allow approval before training. No automatic claim that an output has the correct 'Nagato persona' is made.

## Next step after results

1. If knowledge/persona or Real Persona NLL has regressed materially from v0.10.0, prepare a **balanced mixed SFT curriculum** incorporating Real Persona, source-attributed Nagato QA, Nagato persona and optional short copy tasks without overwhelming any one track.
2. Preserve a **frozen v0.10.0** benchmark checkpoint as reference; validate all four tracks and actual copy-task performance when choosing a combined model.
3. Only after choosing model settings using development results should the original untouched Nagato/Persona final test material be opened for final confirmation.

**Limit:** These validation examples derive from splits used for previous model selection, so they are **not an independent final benchmark**. Exact string match is a strict proxy: accurate paraphrases are judged as failures, and matching source-derived text does not make its factual claims true.
