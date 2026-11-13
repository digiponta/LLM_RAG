# LLM_RAG v0.9.3 — RoPE architecture ablation

## Scientific question
v0.9.2 uses matched 193,636 training pairs and 22,750 test pairs with 256-token windows. The trained Legacy/Legacy NLL = 2.59906 and Structured/Structured = 2.60424 on the same sampled 1,000 held-out IDs. They are nearly equal. Does explicitly encoding relative token positions with RoPE help beyond formatting changes?

## Implementation
- `rope_model_v093.py`: a single-head Q/K rotary attention layer. Applies position rotation to q and k only, retaining the original attention projections, value path, causal mask, model hidden size, vocab, and checkpoint parameter names/shapes.
- Position encodings use deterministic sin/cos with 10,000 base. The optional model configuration contains `position_encoding: rope_v093`; there are no extra trainable parameters.
- `train_real_dialogue_v082.py`: adds `--rope --allow-rope-conversion` to explicitly initialize from an existing non-RoPE checkpoint; the default non-RoPE path remains backward compatible.
- `evaluate_real_dialogue_v082.py`: adds `--baseline-rope` and `--tuned-rope` to load checkpoints with the correct attention implementation. Checkpoints marked RoPE are rejected if loaded as ordinary attention.
- `evaluate_context_swap_v088.py` and `evaluate_context_generation_v091.py`: add `--rope` to inspect the new model using existing held-out diagnostics.
- `tests/test_rope_v093.py`: 12 new tests for position rotation, norm, causality, exact weight transfer, config/checkpoint safeguards, and training gradients.

## On Windows — branch, tests
    cd C:\Users\inoma\git\misc\LLM_RAG
    git fetch origin
    git switch feature/rope-ablation-v0.9.3
    python -m unittest discover -s tests -v

Expected: prior 161 tests + 12 new = 173 tests. This count is not a claim of execution in the Windows environment.

## Experiment B — RoPE + Legacy input (controlled against v0.9.2 Legacy)
    python train_real_dialogue_v082.py `
      --checkpoint model/model-rag-context256-init-v0.9.0.pt `
      --source data/dialogue/real_persona_structured_v092/paired_legacy_train_v092.jsonl `
      --validation data/dialogue/real_persona_structured_v092/paired_legacy_validation_v092.jsonl `
      --max-train 0 --max-validation 0 --epochs 2 --batch-size 8 `
      --learning-rate 0.00005 `
      --rope --allow-rope-conversion `
      --output model/model-rag-rope-legacy-v0.9.3.pt `
      --report results/rope_legacy_train_v093.json --device auto

Train and validation are exactly the same dataset/cohort used by the existing plain Legacy v0.9.2 model, with identical seed, optimizer, checkpoint weights, steps and hyperparameters. The only intended change is rotary Q/K attention.

### Held-out Test (same QA IDs and identical Legacy input)
    python evaluate_real_dialogue_v082.py `
      --baseline model/model-rag-legacy-paired-v0.9.2.pt `
      --tuned model/model-rag-rope-legacy-v0.9.3.pt `
      --tuned-rope `
      --test-data data/dialogue/real_persona_structured_v092/paired_legacy_test_v092.jsonl `
      --sample 1000 --generations 20 --batch-size 8 `
      --output results/rope_legacy_test_v093.json --device auto

Compare answer NLL and full response examples, NOT only perplexity. Test samples are held out by original dialogue-file split; the same speakers/topics and near-duplicates may still occur.

## Experiment C — RoPE + Structured input (independent incremental test)
After inspecting Experiment B, optionally run:
    python train_real_dialogue_v082.py `
      --checkpoint model/model-rag-context256-init-v0.9.0.pt `
      --source data/dialogue/real_persona_structured_v092/paired_structured_train_v092.jsonl `
      --validation data/dialogue/real_persona_structured_v092/paired_structured_validation_v092.jsonl `
      --max-train 0 --max-validation 0 --epochs 2 --batch-size 8 `
      --learning-rate 0.00005 --rope --allow-rope-conversion `
      --output model/model-rag-rope-structured-v0.9.3.pt `
      --report results/rope_structured_train_v093.json --device auto

    python evaluate_real_dialogue_v082.py `
      --baseline model/model-rag-structured-paired-v0.9.2.pt `
      --tuned model/model-rag-rope-structured-v0.9.3.pt `
      --tuned-rope `
      --test-data data/dialogue/real_persona_structured_v092/paired_structured_test_v092.jsonl `
      --sample 1000 --generations 20 --batch-size 8 `
      --output results/rope_structured_test_v093.json --device auto

NOTE: `--baseline-rope=false` is NOT a valid argparse flag. Simply omit `--baseline-rope` when running the above optional evaluation.

## Context swap analysis (optional)
    python evaluate_context_swap_v088.py `
      --checkpoint model/model-rag-rope-legacy-v0.9.3.pt `
      --rope --sample 200 `
      --test-data data/dialogue/real_persona_structured_v092/paired_legacy_test_v092.jsonl `
      --output results/rope_context_swap_v093.json --device auto

## Limitations
- RoPE provides relative token position information, NOT understanding of who 'それ' refers to.
- Rotations are added at the attention computation level; reusing ordinary attention weights is an architecture conversion, not equivalent inference. The new RoPE model MUST be fine-tuned.
- This LLM_GPU v0.3 model is single-head with ~754,560 params. Q/K rotary positions provide extra relational bias but cannot guarantee natural Japanese generation.
- No change to original LLM_GPU repo; all local checkpoints and private dialogue files remain in the user's Windows workspace.
- The current model uses no explicit separate sentence-level/turn-level position encoding. Compare A/B using exact same input form before testing RoPE+Structured.
