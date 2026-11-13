# LLM_RAG v0.9.9 — Unified Real Persona Chat + Nagato SFT

## Goal

Use **the SAME model, tokenizer, question/context/answer prompt, answer-only loss, train loop, validation mechanism, and generation path** for Real Persona Chat dialogues and `data-nagato.txt`. No separate Nagato language model is created. Unlike `chat.py --rag`, this actually fine-tunes the Transformer weights.

## Unified record

Both origins use JSONL records with required fields `id`, `conversation_id`, `question`, `answer` and optional `context`. Both use the existing `compact_prompt`:

    前:<context>      # only when context is present
    問:<question>
    答:<answer to be predicted>

The existing `train_real_dialogue_v082.py` builds exactly the same autoregressive answer-only loss mask for both sources. It uses the existing compatible LLM_GPU tokenizer and v0.9.5 single-head RoPE Medium checkpoint. No new architecture or loss function is introduced.

### Where Nagato records differ in provenance (not in training mechanics)

- The original `data/knowledge/data-nagato.txt` contains statements, opinions, speculative physics, fiction, etc., **not paired user/assistant turns**. It cannot honestly be treated as pre-existing genuine Q/A.
- `prepare_unified_sft_v099.py` creates **two training variants** per source segment, with identical SFT schema: (1) a synthetic source-attributed question with a short quotation cue, whose answer quotes the source inside `長門コーパスには「...」と記されています。`; (2) an in-character Nagato-style prompt with a quotation cue, whose answer is the verbatim original segment. Both retain `review_state=SOURCE_QUOTE_UNVERIFIED`.
- A short quote cue distinguishes different source statements on the same broad topic. In the attributed variant the answer is explicitly attributed; in the in-character variant it is a verbatim authored persona statement, **not independent verified scientific fact or a new paraphrase**. Both share the same original source-line provenance and split.
- Long original lines are segmented to fit the 256-token context. All segments of a single original line stay together in the same train/validation/test split; exact duplicate lines are removed. Near-duplicates in different original lines could still leak.
- Nagato TRAIN is repeated to a tunable mixture fraction (default 5%) because the Real Persona Chat training set is far larger. Nagato VALIDATION and TEST are never oversampled. The default 5% implies extensive repetition of a small corpus, so report overfitting and chat regressions.
- These are source-reproduction and source-derived persona-response training examples. They do **not** by themselves teach universal fact answering, self-identification as 長門有希 in unrelated exchanges, or faithful abstract paraphrase. The source format and quality remain a limitation.

## Step 1 — Checkout + Windows tests

    cd C:\Users\inoma\git\misc\LLM_RAG
    git fetch origin
    git switch feature/unified-persona-nagato-sft-v0.9.9
    python -m unittest discover -s tests -v

Expect previous 250 + 16 new = **266** tests; Windows CUDA training/validation remains to be performed locally.

## Step 2 — Generate unified local-only train and validation data

    python prepare_unified_sft_v099.py `
      --nagato data/knowledge/data-nagato.txt `
      --persona-train data/dialogue/real_persona_structured_v092/paired_legacy_train_v092.jsonl `
      --persona-validation data/dialogue/real_persona_structured_v092/paired_legacy_validation_v092.jsonl `
      --nagato-fraction 0.05 `
      --max-chars 90 `
      --context-length 256 `
      --output-dir data/dialogue/unified_v099

Creates local-only:

    data/dialogue/unified_v099/unified_train_v099.jsonl
    data/dialogue/unified_v099/unified_validation_v099.jsonl
    data/dialogue/unified_v099/nagato_train_v099.jsonl
    data/dialogue/unified_v099/nagato_validation_v099.jsonl
    data/dialogue/unified_v099/nagato_test_v099.jsonl
    data/dialogue/unified_v099/unified_report_v099.json

All source text in these JSONL files stays local. The short console report includes only counts, not private dialogue. Do NOT commit any generated JSONL or checkpoint.

## Step 3 — Fine-tune shared RoPE Medium on BOTH sources

    python train_real_dialogue_v082.py `
      --checkpoint model/model-rag-rope-medium-v0.9.5.pt `
      --source data/dialogue/unified_v099/unified_train_v099.jsonl `
      --validation data/dialogue/unified_v099/unified_validation_v099.jsonl `
      --max-train 0 --max-validation 0 `
      --epochs 1 --batch-size 4 --learning-rate 0.00002 `
      --rope --device auto `
      --output model/model-rag-unified-nagato-v0.9.9.pt `
      --report results/unified_nagato_train_v099.json

Both sources are tokenized and optimized in the **same** DataLoader, optimizer, and answer-token cross entropy. Baseline checkpoint is read-only. Because oversampled Nagato records share a `conversation_id`, there must be no leakage into validation and the mixture checker verifies this.

## Step 4 — Evaluate Nagato held-out lines (without training leakage)

    python evaluate_real_dialogue_v082.py `
      --baseline model/model-rag-rope-medium-v0.9.5.pt --baseline-rope `
      --tuned model/model-rag-unified-nagato-v0.9.9.pt --tuned-rope `
      --test-data data/dialogue/unified_v099/nagato_test_v099.jsonl `
      --sample 100 --generations 20 --batch-size 4 `
      --output results/unified_nagato_test_v099.json --device auto

Small Nagato test counts may be fewer than 100; the evaluator selects up to available rows. Also compare on the **untouched original** Real Persona Chat test split to check regressions:

    python evaluate_real_dialogue_v082.py `
      --baseline model/model-rag-rope-medium-v0.9.5.pt --baseline-rope `
      --tuned model/model-rag-unified-nagato-v0.9.9.pt --tuned-rope `
      --test-data data/dialogue/real_persona_structured_v092/paired_legacy_test_v092.jsonl `
      --sample 1000 --generations 20 --batch-size 4 `
      --output results/unified_persona_test_v099.json --device auto

Do NOT compare Nagato and Persona NLL numerically as if they are the same task; compare before/after within each split separately. Short, attributed-source answer reproduction is not factuality or faithful dialogue understanding.

## Step 5 — Chat on the same unified model

    python chat.py `
      --checkpoint model/model-rag-unified-nagato-v0.9.9.pt `
      --device auto --repetition-penalty 1.0

Optional retrieval remains independent:

    python chat.py `
      --rag --docs data/knowledge `
      --checkpoint model/model-rag-unified-nagato-v0.9.9.pt `
      --device auto

**Important:** Chat with the generic question `宇宙とは` is NOT trained to reproduce some one of many different source statements: the generated Nagato SFT prompts ask source-attributed, quote-cued questions. Evaluate those exact prompt types first, then measure transfer to free-form dialogue separately.

## Operational boundaries

- The new generated mixture and results must stay on your computer. `.gitignore` now excludes the `data/dialogue/`, `results/` and `model/*.pt` directories; gitignore does NOT untrack already committed files.
- Original Real Persona Chat source files are never rewritten. The original Nagato text is read-only.
- A checkpoint cannot be assumed to 'know' statements simply because its NLL decreases. Track first-token accuracy, generated text and unknown/unsupported responses, and verify whether everyday dialogue worsens.
- This update intentionally does not rewrite `LLM_GPU` or install an external classifier. The goal is one SFT pipeline for two kinds of content, with provenance kept explicit.
