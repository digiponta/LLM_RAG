# LLM_RAG v0.10.5 — Position Learning

## Motivation and experimental boundaries

In v0.10.4, frozen v0.10.0 and v0.10.3 checkpoints produced 0/180 *complete* one-character replies, despite v0.10.3 achieving 24/180 correct first answer tokens. Changing evidence changed the free output in 90/90 pairs for the v0.10.3 model, but changing output is not itself evidence of correct position extraction.

v0.10.5 trains **explicit position instruction following** with the existing 2.1M parameter single-head RoPE Medium checkpoint. It changes neither architecture nor tokenizer. It is an *isolated position experiment*, not a release of a more capable conversational model.

## 1. Checkout and tests

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/position-learning-v0.10.5
python -m unittest discover -s tests -v
```

Expected **366 tests** (349 previous + 17 new); Windows CUDA execution remains to be done locally.

## 2. Generate position SFT

```powershell
python prepare_position_learning_v105.py `
    --copy-train data/dialogue/copy_general_v103/copy_general_train_v103.jsonl `
    --copy-development data/dialogue/copy_general_v103/copy_general_development_v103.jsonl `
    --positions 1 5 10 `
    --output-dir data/dialogue/position_v105
```

Each 10-character row is transformed into three `question/context/answer` SFT rows: '資料の1文字目...', '資料の5文字目...', and '資料の10文字目...'. Only the selected **single Unicode character** is the answer. Source-provenance and natural/synthetic labels are retained.

Expected upper bound: **6,000 TRAIN** and **600 DEVELOPMENT** rows. A source whose target at ANY of the three positions is tokenizer unknown, or whose full prompt is over-context, is excluded **entirely** from the three-position group; check actual counts in the report. The existing v0.10.3 TRAIN vs DEVELOPMENT separation remains intact; the final Nagato TEST is not touched. Locally generated datasets remain gitignored.

## 3. Fine-tune with existing shared masked SFT

**Important:** The default v0.8.2 trainer discards 1-token answers. v0.10.5 introduces `--min-answer-tokens 1` as an **opt-in**; its default remains 2 so older dialogue training is unchanged.

```powershell
python train_real_dialogue_v082.py `
    --checkpoint model/model-rag-copy-general-v0.10.3.pt `
    --source data/dialogue/position_v105/position_train_v105.jsonl `
    --validation data/dialogue/position_v105/position_development_v105.jsonl `
    --min-answer-tokens 1 `
    --max-train 0 --max-validation 0 `
    --epochs 5 --batch-size 16 --learning-rate 0.00005 `
    --rope --device auto `
    --output model/model-rag-position-v0.10.5.pt `
    --report results/position_train_v105.json
```

The shared trainer still applies answer-only masked cross-entropy, including **EOS after the target character**, and saves only the checkpoint achieving a better development NLL. If it reports that no checkpoint was saved, inspect `results/position_train_v105.json` instead of proceeding with a missing file.

## 4. Compare free generation, EOS, and counterfactual tracking

```powershell
python evaluate_position_learning_v105.py `
    --baseline model/model-rag-copy-general-v0.10.3.pt `
    --tuned model/model-rag-position-v0.10.5.pt `
    --dataset data/dialogue/copy_general_v103/copy_general_development_v103.jsonl `
    --positions 1 5 10 --sample 30 `
    --max-new-tokens 4 --device auto `
    --output results/position_generalization_v105.json
```

Report includes 180 generated cases (30 source rows × 3 positions × original/swapped) when all source rows qualify. The evaluator compares **frozen** v0.10.3 and v0.10.5 on identical prompts, and separates:

- **Exact single-character free generation**: output consists of the target and nothing else.
- **First target token accuracy**: answer-token selection at the first generation step.
- **Gold-conditioned EOS probability**: teacher-force the correct answer and measure P(EOS | prompt, correct answer); not the same as successful free-generation termination.
- **Free-generation EOS rate** and average output length.
- **Counterfactual pair correctness**: both original and swapped evidence produce their corresponding gold character; also reported **conditioned on cases where the correct character actually changes**.
- Per-position, natural/synthetic and original/swapped breakdowns.

Inspect aggregates without private text:

```powershell
$r = Get-Content results/position_generalization_v105.json -Raw -Encoding UTF8 | ConvertFrom-Json
$r.groups | ConvertTo-Json -Depth 8
$r.counterfactuals | ConvertTo-Json -Depth 8
```

For manual examples use `--include-text` and a DIFFERENT `--output` path; keep that local and do not commit it.

## Interpretation

| Outcome | Meaning / next step |
|---|---|
| First-token accuracy rises but exact remains near zero | EOS termination/continuation likely needs improvement; consult EOS metrics |
| Position 1 improves, 5 and 10 stay low | Position-conditioned target selection remains inadequate |
| Only natural text improves | Model may depend on language patterns rather than position transfer |
| Output swap rate rises but paired correct tracking remains low | Input-sensitive but still not performing correct extraction |
| All three positions and both text categories improve | Test 30/50-character position tasks, then compare copy-only vs copy-pointer baselines |

These are **development** findings, not independent final heldout results. This experiment does not establish fact recall, retrieval or evidence-grounded answer composition. Existing v0.10.0 remains the general chat checkpoint; v0.10.5 is research-only.
