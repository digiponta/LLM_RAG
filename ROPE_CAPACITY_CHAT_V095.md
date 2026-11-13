# LLM_RAG v0.9.5 — RoPE capacity study + chat.py

## Context
v0.9.3 showed 2.59906 -> 2.49152 held-out answer NLL after RoPE introduction on 1,000 matched examples. v0.9.4 showed 20-example RoPE advantage on 18/20 cases, but many responses were generic, irrelevant or ungrammatical. This release adds a local interactive CLI and experimental capacity profiles; neither implies demonstrated production quality.

## 1. Install/test (Windows PowerShell)
    cd C:\Users\inoma\git\misc\LLM_RAG
    git fetch origin
    git switch feature/rope-capacity-chat-v0.9.5
    python -m unittest discover -s tests -v

15 new tests + existing 185 => expected 200 tests. Windows/CUDA execution is pending.

## 2. Run interactive Chat.py immediately with trained RoPE checkpoint
    python chat.py --checkpoint model/model-rag-rope-legacy-v0.9.3.pt --device auto

Example:
    You> こんにちは
    AI> ...
    You> 読書は好きですか？
    AI> ...
    You> /status
    You> /reset
    You> /exit

Use a different existing legacy checkpoint (architecture auto-detected):
    python chat.py --checkpoint model/model-rag-legacy-paired-v0.9.2.pt --device auto

Local document retrieval is optional (not the model's training prompt; NOT verified evidence):
    python chat.py --rag --docs data/knowledge --checkpoint model/model-rag-rope-legacy-v0.9.3.pt --device auto

Extra commands with --rag: /reload (rescan documents), /sources (show source filenames). RAG results are inserted as short `資料:` text in the existing compact `前:` history prompt. The model was trained as dialogue SFT, NOT to consistently ground answers in retrieved documents; generated text can ignore or misstate retrieved evidence. The old independent `rag_persona.py` remains available with its custom persona; `chat.py` does not impersonate that persona by default.

Chat controls:
- `--encoding auto|legacy|rope`: auto-detect checkpoint marker (RoPE v0.9.3), refuse mismatched explicit selection.
- `--temperature 0` default greedy (deterministic); positive temperatures use top-k random sampling; `--seed` controls sampling.
- `--max-new-tokens 48`, `--repetition-penalty 1.3`, `--verbose`: output controls.
- Session keeps the last two complete utterances and drops whole past utterances, then shortens optional retrieved evidence, when near context limit. It never silently truncates the current question. No chat history is persisted to disk.

## 3. Compare model capacity fairly
All capacity variants use same RoPE code, tokenizer, context length 256 and the existing EXACT SAME data partition. Crucially, changing d_model or layer count makes pretrained v0.9.3 weights incompatible; this first capacity experiment uses fresh initialization with a known seed for EACH profile. The previously fine-tuned 754,560-parameter model is a useful production-style reference, but not a fair from-scratch capacity control.

Small (64x2x256; 754,560 parameters):
    python initialize_capacity_v095.py --profile small --checkpoint model/model-rag-rope-small-init-v0.9.5.pt --report results/rope_small_init_v095.json

Medium (128x4x512; expect ~2M parameters, exact count reported):
    python initialize_capacity_v095.py --profile medium --checkpoint model/model-rag-rope-medium-init-v0.9.5.pt --report results/rope_medium_init_v095.json

Optional Large (128x6x512; exact count reported):
    python initialize_capacity_v095.py --profile large --checkpoint model/model-rag-rope-large-init-v0.9.5.pt --report results/rope_large_init_v095.json

Train scratch Small (same SFT data for all profiles):
    python train_real_dialogue_v082.py `
      --checkpoint model/model-rag-rope-small-init-v0.9.5.pt `
      --source data/dialogue/real_persona_structured_v092/paired_legacy_train_v092.jsonl `
      --validation data/dialogue/real_persona_structured_v092/paired_legacy_validation_v092.jsonl `
      --max-train 0 --max-validation 0 --epochs 2 --batch-size 4 --learning-rate 0.00005 --rope `
      --output model/model-rag-rope-small-v0.9.5.pt --report results/rope_small_train_v095.json --device auto

Train scratch Medium:
    python train_real_dialogue_v082.py `
      --checkpoint model/model-rag-rope-medium-init-v0.9.5.pt `
      --source data/dialogue/real_persona_structured_v092/paired_legacy_train_v092.jsonl `
      --validation data/dialogue/real_persona_structured_v092/paired_legacy_validation_v092.jsonl `
      --max-train 0 --max-validation 0 --epochs 2 --batch-size 4 --learning-rate 0.00005 --rope `
      --output model/model-rag-rope-medium-v0.9.5.pt --report results/rope_medium_train_v095.json --device auto

Test on same 1,000 IDs with both RoPE checkpoints:
    python evaluate_real_dialogue_v082.py `
      --baseline model/model-rag-rope-small-v0.9.5.pt --baseline-rope `
      --tuned model/model-rag-rope-medium-v0.9.5.pt --tuned-rope `
      --test-data data/dialogue/real_persona_structured_v092/paired_legacy_test_v092.jsonl `
      --sample 1000 --generations 20 --batch-size 4 `
      --output results/rope_capacity_test_v095.json --device auto

Chat with Medium after training:
    python chat.py --checkpoint model/model-rag-rope-medium-v0.9.5.pt --device auto

## Interpretation and limitations
- Compare NLL and qualitatively reviewed generated Japanese; exact response match is not an adequate dialogue-quality metric.
- Same dataset and epochs control exposure, NOT identical FLOPs, compute time, or architecture-specific initialization dynamics. Log GPU peak memory and wall-clock time when possible.
- A fresh large checkpoint trained for two epochs may underperform a previously fine-tuned smaller model. Compare against the fresh small control to isolate capacity.
- New models are from scratch and may need more training or alternative learning rates; do not silently change hyperparameters between variants.
- LLM_GPU v0.3 attention is single-head, character-level. More parameters alone may not solve reply relevance.
- RAG documents and real persona dialogues are not assumed verified; keep personal conversation JSONL, text-bearing reports and checkpoint files local, not in GitHub commits.
