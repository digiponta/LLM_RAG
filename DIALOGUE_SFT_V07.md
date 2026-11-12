# LLM_RAG v0.7 — Short-Answer Dialogue Fine-Tuning

## Goal

Run a controlled supervised fine-tuning (SFT) experiment using the 271
synthetic quotation-style QA examples prepared by v0.6 (230 train / 41 validation).
Keep the audit-free context-aware extractive runtime of v0.6 unchanged.

## Model constraint

The inspected LLM_GPU v0.3 implementation has a character-based tokenizer,
~755k parameters, and a 64-token context window. Full long quotations
cannot be processed as a coherent question + answer sequence.

`train_dialogue_v07.py` takes one sentence from each source quotation,
preferring a sentence containing the source topic. It uses a compact
`問:<topic>\n答:` prefix and masks prefix targets with -100 so that
only answer and optional EOS tokens contribute to the SFT objective.
Long answers are cut to the available space within the model context.
Rows are skipped if they cannot fit enough answer tokens or their
unknown-token rate exceeds the configured threshold.

Train/validation IDs must be disjoint. The source checkpoint, tokenizer,
training file and validation file remain unchanged. Separate output files
and a non-overwrite guard protect existing artifacts.

**Important:** These 41 validation rows were generated from the SAME Nagato
corpus. The model was already fine-tuned on that corpus in v0.2, so
this is NOT an independent holdout or proof of generalization.

## 1. Checkout and tests (PowerShell)

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/dialogue-sft-v0.7
python -m unittest discover -s tests -v
```

Nine new CPU-only unit tests are added. The prior branch passed 42 tests;
51 tests are expected if no other tests have been added.

## 2. Train (CUDA / CPU automatic)

Use the v0.6 dataset files already generated locally:

```powershell
python train_dialogue_v07.py `
    --gpu-dir ../LLM_GPU `
    --checkpoint model/model-rag-nagato-v0.2.pt `
    --tokenizer ../LLM_GPU/model/tokenizer.json `
    --train-data data/dialogue/nagato_synthetic_train_v06.jsonl `
    --val-data data/dialogue/nagato_synthetic_val_v06.jsonl `
    --output model/model-rag-dialogue-v0.7.pt `
    --report results/dialogue_sft_v07.json `
    --epochs 5 `
    --batch-size 16 `
    --learning-rate 0.00005 `
    --device auto
```

The best checkpoint is saved ONLY if validation answer NLL improves
relative to the starting model. If no epoch improves it, there will be
a report but no v0.7 checkpoint. Do not interpret improvement in this
training metric as evidence of improved open-domain dialogue.

## 3. Compare generations

```powershell
python compare_dialogue_v07.py `
    --gpu-dir ../LLM_GPU `
    --tokenizer ../LLM_GPU/model/tokenizer.json `
    --baseline model/model-rag-nagato-v0.2.pt `
    --tuned model/model-rag-dialogue-v0.7.pt `
    --output results/dialogue_compare_v07.json `
    --device auto
```

The comparison uses deterministic generation (temperature 0), same
compact prompt, and the same set of topic probes. Its automatic scores
(nonempty, Japanese present, topic present) are diagnostic ONLY, not
valid measures of coherence, grounding or scientific correctness.
Read and compare the output text manually.

## 4. Current safe runtime

```powershell
python rag_persona_v06.py --docs data/knowledge --mode extractive
```

v0.7 does not auto-switch the user-facing persona runtime to the
experimental SFT checkpoint. Until Japanese generation becomes coherent,
extractive v0.6 remains the default.

## Planned v0.8

Create a truly independent, new set of conversational examples
with topic switches, pronoun reference, unknown questions, and spontaneous
dialogue. Then decide whether to expand context and model capacity or
retain a hybrid retrieval-first responder.
