# LLM_RAG v0.9.6 — Multi-Head Attention + RoPE controlled ablation

## Purpose
v0.9.5 RoPE Medium (d_model=128, four layers, feed-forward 512, one head, 2,101,248 parameters) achieved Test answer NLL **2.3489517906** on the fixed 1,000 held-out dialogue examples after two scratch-training epochs. Its interactive chat still shows significant topic drift. Test whether four rotary Q/K heads improve predictions and generated replies *without changing parameter count, embedding dimensions, training rows or optimizer*.

## Architecture
- The original single-head implementation projects Q/K/V to [B,T,128] and computes scaled dot-product Attention on all 128 channels (RoPE over 128).
- New four-head implementation projects Q/K/V to the SAME [B,T,128], splits into [B,4,T,32], applies RoPE separately to the Q/K of each head, scales each attention score by sqrt(32), and merges the heads before the existing output projection.
- All parameter keys, shapes and values can be copied unchanged from the **fresh**, untrained single-head Medium init. Forward outputs are NOT the same, so the MHA checkpoint requires its own training.
- `position_encoding=rope_v096_mha` and `num_heads=4` mark MHA checkpoints. The loader refuses accidental single-head interpretation or incompatible head counts.
- This is standard multi-head attention behavior, not a trainable speaker-identity mechanism. It does not ensure conversation semantics.

## Windows PowerShell: install and test
    cd C:\Users\inoma\git\misc\LLM_RAG
    git fetch origin
    git switch feature/multihead-rope-v0.9.6
    python -m unittest discover -s tests -v

Expected 200 previous + 16 new = 216 tests; the actual count and CUDA results must be confirmed on Windows. Do not merge until tests pass.

## Step 1: Prepare four-head Medium initial checkpoint (same initial weights)
    python initialize_mha_v096.py `
      --source model/model-rag-rope-medium-init-v0.9.5.pt `
      --output model/model-rag-mha4-medium-init-v0.9.6.pt `
      --report results/mha4_medium_init_v096.json `
      --heads 4

Only epoch=0 untrained RoPE checkpoints can be converted by this utility. NEVER replace the original checkpoint or convert the already trained `model-rag-rope-medium-v0.9.5.pt` to MHA for an accuracy comparison.

## Step 2: Train 4-head model on EXACTLY the same corpus as 1-head Medium
    python train_real_dialogue_v082.py `
      --checkpoint model/model-rag-mha4-medium-init-v0.9.6.pt `
      --source data/dialogue/real_persona_structured_v092/paired_legacy_train_v092.jsonl `
      --validation data/dialogue/real_persona_structured_v092/paired_legacy_validation_v092.jsonl `
      --max-train 0 --max-validation 0 `
      --epochs 2 --batch-size 4 --learning-rate 0.00005 `
      --rope --mha --mha-heads 4 `
      --output model/model-rag-mha4-medium-v0.9.6.pt `
      --report results/mha4_medium_train_v096.json --device auto

Do not pass `--allow-mha-conversion` when the input checkpoint is already marked MHA. It is necessary only for explicit conversion from a one-head RoPE checkpoint. Avoid accidental overwrites.

## Step 3: Matched held-out Test NLL (1,000 answers)
    python evaluate_real_dialogue_v082.py `
      --baseline model/model-rag-rope-medium-v0.9.5.pt --baseline-rope `
      --tuned model/model-rag-mha4-medium-v0.9.6.pt --tuned-rope --tuned-mha `
      --mha-heads 4 `
      --test-data data/dialogue/real_persona_structured_v092/paired_legacy_test_v092.jsonl `
      --sample 1000 --generations 20 --batch-size 4 `
      --output results/mha4_medium_test_v096.json --device auto

Baseline result to compare: one-head RoPE Medium `answer_token_nll = 2.3489517906`. The new model should not be declared better unless the comparison shows it.

## Step 4: Chat with MHA Medium
    python chat.py --checkpoint model/model-rag-mha4-medium-v0.9.6.pt --device auto

Chat auto-detects `rope_v096_mha` and prints `Multi-Head RoPE (4 heads)`. Existing `--encoding auto|legacy|rope` remains supported and `--encoding mha` is additionally available. Both RAG-off and optional `--rag` modes remain available.

## Step 5: Same-history generation review (optional; local private text)
    python evaluate_rope_generation_v094.py `
      --baseline model/model-rag-rope-medium-v0.9.5.pt --baseline-rope `
      --rope-checkpoint model/model-rag-mha4-medium-v0.9.6.pt --mha --mha-heads 4 `
      --test-data data/dialogue/real_persona_structured_v092/paired_legacy_test_v092.jsonl `
      --sample 20 --include-text `
      --output results/mha4_generation_text_v094.json --device auto

This reuses the v0.9.4 report schema (including the `_text_v094.json` filename requirement). Text-bearing reports contain private conversations and MUST remain on the local machine, not GitHub.

## Interpretation
Same 193,636 train and 23,055 validation questions, context 256, tokenizer vocabulary 5,117, seed 42, learning rate 5e-5, batch 4, 2 epochs, and parameter count. This isolates the effect of head partitioning and separate per-head RoPE under a fixed optimization schedule. The calculation graph and per-head score scaling change, so training trajectories are not identical. MHA may improve, match or worsen NLL and dialogue quality; no outcome is assumed in advance.

Follow with manual dialogue relevance and grammaticality review, and eventually multiple random seeds to understand run-to-run variability.
