# LLM_RAG v0.10.2 — Small Copy-Learning Probe

## Motivation

The v0.10.1 evaluation of frozen v0.9.9 and v0.10.0 showed 0/36 exact copying and 0/36 first-token accuracy for both models. Manual examples revealed replies such as 「そうなんですね」 and 「なるほど」 instead of supplied source strings. The smallest useful next experiment is a **train-set overfit probe**, not another large 203,827-row SFT.

**Question A:** Can the *existing* ~2.1M parameter single-head RoPE Medium LLM learn to copy 10 exact characters from source text that it has seen during training?

**Question B:** Does this work on a separate 10-character development set drawn from unseen original source lines?

An improvement on training examples alone is memorization, **not** copying generalization. The script tracks both. No modifications to Persona training data, existing checkpoints, or final Nagato test data.

## 1. Update and test

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/copy-learning-probe-v0.10.2
python -m unittest discover -s tests -v
```

Expected 300 existing + 15 new = **315 tests**; these have **not yet been run** in the user's local CUDA environment.

## 2. Prepare source-disjoint copy data

```powershell
python prepare_copy_learning_v102.py `
    --width 10 `
    --train-count 16 `
    --development-count 8 `
    --output-dir data/dialogue/copy_probe_v102
```

The train file comes only from `unified_v099/nagato_train_v099.jsonl`. Development comes only from `unified_v099/nagato_validation_v099.jsonl`. The **Nagato TEST split remains sealed**. No original source line appears in both training and development. Each answer equals exactly the 10-character `資料:` context suffix, not a generative paraphrase. The full prompt and response are accepted under the existing 256-token, low-UNK SFT encoder.

Local output:
- `data/dialogue/copy_probe_v102/copy_train_v102.jsonl`
- `data/dialogue/copy_probe_v102/copy_development_v102.jsonl`
- `data/dialogue/copy_probe_v102/copy_prepare_report_v102.json`

The builder never overwrites existing data unless explicitly given `--overwrite`.

## 3. Micro-overfit study (frozen original checkpoint preserved)

```powershell
python train_copy_learning_v102.py `
    --checkpoint model/model-rag-binding-v0.10.0.pt `
    --train data/dialogue/copy_probe_v102/copy_train_v102.jsonl `
    --development data/dialogue/copy_probe_v102/copy_development_v102.jsonl `
    --epochs 40 `
    --eval-every 5 `
    --batch-size 4 `
    --learning-rate 0.0002 `
    --max-new-tokens 40 `
    --device auto `
    --output model/model-rag-copy-probe-v0.10.2.pt `
    --report results/copy_learning_probe_v102.json
```

This training task uses **the same tokenizer, RoPE architecture, AdamW, masked answer-token cross-entropy, gradient clipping, and prompt schema** as the existing unified SFT pipeline, but isolates 16 examples for a learning-capability diagnostic. It prints and stores **baseline and per-epoch** training and development NLL, exact-copy counts, first-token accuracy, prefix coverage, position accuracy and mean generated length. All generation uses greedy decoding and repetition penalty 1.0.

**Important:** Unlike the general SFT trainer that saves only when validation NLL improves, this controlled probe saves the **final model even when development gets worse**, precisely to reveal overfitting. It must not be released as a better general conversational model.

## 4. Read summary

```powershell
$r = Get-Content results/copy_learning_probe_v102.json -Raw -Encoding UTF8 | ConvertFrom-Json
$r.before | ConvertTo-Json -Depth 5
$r.history | Select-Object epoch,train_nll,development_nll,
    @{N="train_exact";E={$_.train_generation.exact}},
    @{N="dev_exact";E={$_.development_generation.exact}},
    @{N="train_first";E={$_.train_generation.first_token}},
    @{N="dev_first";E={$_.development_generation.first_token}} |
    Format-Table -AutoSize
```

## Decision tree

| Observation | Interpretation | Next direction |
|---|---|---|
| 0/16 even on memorized TRAIN, train NLL stays high | Basic copy relation not learned within budget | Inspect tokenization, answer mask, learning rate and greedy decoding; check optimizer dynamics |
| TRAIN copying improves, DEVELOPMENT remains 0/8 | Training-set memorization without transfer | Broaden variety of copying tasks, regularize and re-evaluate |
| TRAIN and DEVELOPMENT 10-character copying improve | Short copy instruction generalizes to source-disjoint lines | Expand to 30/50 characters before reintegrating Persona |
| DEVELOPMENT NLL improves but exact stays 0/8 | Teacher-forced gains still fail free generation | Inspect first-token ranks, EOS and decoded outputs; do not claim success |

All data remains local, ignored under `data/dialogue/`, `model/*.pt`, and `results/`. Do not commit or redistribute original Nagato or private Persona content. The final Nagato test split should be used **only after** choosing the copy-training procedure based on training and development results.
