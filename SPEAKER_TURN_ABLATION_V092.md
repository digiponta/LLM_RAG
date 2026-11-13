# LLM_RAG v0.9.2 — Speaker/Turn Prompt Ablation

## Purpose
v0.9.1: 20 examples, context changes 95% of greedy outputs, mean answer NLL advantage +0.0973, but output semantics and grammar remain unreliable. v0.9.2 tests whether explicit speaker and turn boundaries help on held-out QA, WITHOUT modifying attention/position weights.

## Prompt designs
Legacy (already used in v0.9.0):
    前:寒いですね／暖かい服を着ます
    問:あなたは？
    答:

Structured (new; chronology explicit):
    B:寒いですね
    A:暖かい服を着ます
    B:あなたは？
    A:

A=answering speaker and B=questioning speaker for EACH example. These are RELATIVE roles, not stable identities and not a persona. The original dialogue JSON is required to recover the precise speakers of historical turns. Existing QA text, answer token masking, context length, checkpoint structure and tokenizer are unchanged. The prompt override is supported only when the row explicitly contains a validated `prompt_override` field; legacy prompts continue to work without change.

## Stage 1 — fetch and validate
    cd C:\Users\inoma\git\misc\LLM_RAG
    git fetch origin
    git switch feature/speaker-turn-ablation-v0.9.2
    python -m unittest discover -s tests -v

## Stage 2 — build exactly paired data on LOCAL original JSON
    python prepare_speaker_turn_v092.py --context-length 256 --turns 2 --base-dir data/dialogue/real_persona_context256_v090 --output-dir data/dialogue/real_persona_structured_v092

The converter loads the existing v0.9.0 context256 dialogue-file-separated splits, reconstructs verified history from the original local source JSON, and retains the exact same QA ID in BOTH variants ONLY IF both tokenized prompts/answer fit 256 with acceptable UNK rate. Report: data/dialogue/real_persona_structured_v092/paired_report_v092.json. New cohort size will be BELOW or equal to 193,636 Train cases; do not assume full retention.

## Stage 3 — train two models from identical frozen initial checkpoint
Baseline legacy:
    python train_real_dialogue_v082.py --checkpoint model/model-rag-context256-init-v0.9.0.pt --source data/dialogue/real_persona_structured_v092/paired_legacy_train_v092.jsonl --validation data/dialogue/real_persona_structured_v092/paired_legacy_validation_v092.jsonl --max-train 0 --max-validation 0 --epochs 2 --batch-size 8 --learning-rate 0.00005 --output model/model-rag-legacy-paired-v0.9.2.pt --report results/legacy_paired_train_v092.json --device auto

Structured speaker/turn:
    python train_real_dialogue_v082.py --checkpoint model/model-rag-context256-init-v0.9.0.pt --source data/dialogue/real_persona_structured_v092/paired_structured_train_v092.jsonl --validation data/dialogue/real_persona_structured_v092/paired_structured_validation_v092.jsonl --max-train 0 --max-validation 0 --epochs 2 --batch-size 8 --learning-rate 0.00005 --output model/model-rag-structured-paired-v0.9.2.pt --report results/structured_paired_train_v092.json --device auto

Do NOT compare only their training NLLs: input representations differ. Evaluate both against the same held-out ANSWER tokens using exactly matched question IDs.

## Stage 4 — 2x2 matched Test evaluation
Legacy-input comparison:
    python evaluate_real_dialogue_v082.py --baseline model/model-rag-legacy-paired-v0.9.2.pt --tuned model/model-rag-structured-paired-v0.9.2.pt --test-data data/dialogue/real_persona_structured_v092/paired_legacy_test_v092.jsonl --sample 1000 --generations 20 --batch-size 8 --output results/role_legacy_test_v092.json --device auto

Structured-input comparison:
    python evaluate_real_dialogue_v082.py --baseline model/model-rag-legacy-paired-v0.9.2.pt --tuned model/model-rag-structured-paired-v0.9.2.pt --test-data data/dialogue/real_persona_structured_v092/paired_structured_test_v092.jsonl --sample 1000 --generations 20 --batch-size 8 --output results/role_structured_test_v092.json --device auto

Both commands use the same seed/row order, so paired test IDs are identical. This is a two-model/two-format diagnostic; a model that benefits only on its own prompt format does not necessarily have stronger conversation understanding.

## Limits and safeguards
- Existing LLM_GPU v0.3 has no positional embeddings; adding 'A:'/'B:' text is not a learnable positional representation.
- No new tokenizer/special token IDs: labels are ASCII characters and may be out of vocabulary; the converter enforces existing UNK criteria. Inspect the tokenization and source before interpreting findings.
- Roles are relative per example. Dialogue source has two interlocutors; no stable cross-conversation persona identity is inferred.
- The cohort excludes examples that overflow in either variant. Compare rates with the original 193,636, 23,055, and 22,750 counts.
- Text-bearing local training examples contain private Real Persona Chat conversations. Do not commit JSONL, outputs containing private text, or checkpoints.
- Long-utterance recovery, genuine per-turn role embeddings, learned positional encoding/RoPE, and capacity increases require separately controlled experiments.
