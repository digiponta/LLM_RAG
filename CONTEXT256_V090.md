# LLM_RAG v0.9.0 — Context 256: equal-weight capacity-control experiment

## Architecture review
The current digiponta/LLM_GPU model.py (v0.3) uses token embedding, causal single-head attention and MLP blocks but NO learned or sinusoidal position embeddings. Its context_length is retained in checkpoint config and used by generation and by the RAG dataset encoder; changing this field alone does not change trainable parameter count. It does not make a long-context semantic model automatically.

## Phase 1 — transfer exactly the same v0.8.4 weights
    cd C:\Users\inoma\git\misc\LLM_RAG
    git fetch origin
    git switch feature/context256-v0.9.0
    python -m unittest discover -s tests -v
    python extend_context_v090.py --checkpoint model/model-rag-real-dialogue-full-v0.8.4.pt --context-length 256 --output model/model-rag-context256-init-v0.9.0.pt --report results/context256_init_v090.json

Read the report: parameter_count_old == parameter_count_new and weights_identical == true. Conversion neither trains nor overwrites the v0.8.4 checkpoint.

## Phase 2 — paired history data with token limit 256
The existing v0.8.7 converter is reused with the SAME original v0.8.1 dialogue-file train/validation/test split and authentic 1–2 preceding turns. It rejects truncated and invalid examples. With context length 256, the paired cohort is expected to be larger than at 64, but the precise acceptance count MUST be measured.
    python prepare_context_dialogue_v087.py --context-length 256 --turns 2 --output-dir data/dialogue/real_persona_context256_v090

Inspect data/dialogue/real_persona_context256_v090/context_report_v087.json. This converter reads only the short-turn v0.8.1 197,595 train pairs. Therefore it DOES NOT recover originally excluded long utterances from the 13,583 raw conversation files. That is a separate data-conversion experiment for v0.9.1.

## Phase 3 — matched plain/context training, equal size, same initial weights
Use batch size 8 or 4 on RTX 3070 Ti if CUDA out of memory occurs. Training longer sequences increases attention cost and padding overhead substantially. Do not overwrite checkpoints.

Plain training:
    python train_real_dialogue_v082.py --checkpoint model/model-rag-context256-init-v0.9.0.pt --source data/dialogue/real_persona_context256_v090/paired_plain_train_v087.jsonl --validation data/dialogue/real_persona_context256_v090/paired_plain_validation_v087.jsonl --max-train 0 --max-validation 0 --epochs 2 --batch-size 8 --learning-rate 0.00005 --output model/model-rag-context256-plain-v0.9.0.pt --report results/context256_plain_train_v090.json --device auto

Context training:
    python train_real_dialogue_v082.py --checkpoint model/model-rag-context256-init-v0.9.0.pt --source data/dialogue/real_persona_context256_v090/paired_context_train_v087.jsonl --validation data/dialogue/real_persona_context256_v090/paired_context_validation_v087.jsonl --max-train 0 --max-validation 0 --epochs 2 --batch-size 8 --learning-rate 0.00005 --output model/model-rag-context256-history-v0.9.0.pt --report results/context256_history_train_v090.json --device auto

## Phase 4 — identical held-out questions
Evaluate both models on the SAME context-bearing test rows:
    python evaluate_real_dialogue_v082.py --baseline model/model-rag-context256-plain-v0.9.0.pt --tuned model/model-rag-context256-history-v0.9.0.pt --test-data data/dialogue/real_persona_context256_v090/paired_context_test_v087.jsonl --sample 1000 --generations 20 --batch-size 8 --output results/context256_context_test_v090.json --device auto

Repeat on paired_plain_test_v087.jsonl with a DIFFERENT output name; optionally run evaluate_context_swap_v088.py on the 256 contextual test set with the context-trained model. The 256-token cohort is different from 64's cohort: DO NOT directly attribute changes in overall NLL to context window size. For a strict 64/256 comparison, re-evaluate the 256 model on the shared 64 paired test IDs and their identical tokenized prompt strings.

## Technical cautions
- No position embeddings in LLM_GPU v0.3. Attention can see tokens but lacks explicit sequential position signals, potentially limiting meaningful ordering over long dialogs.
- Model capacity remains fixed but computational memory/time does not: self-attention scales quadratically with sequence length.
- v0.8.4 was trained on short contexts. Expand-only initialization does not improve responses until further training, and improvements are not guaranteed.
- Dialogue-file splits may still share speakers/topics or near-duplicates; ensure privacy of original chats and generated datasets.
- Do not commit real dialogue JSONL, checkpoints, or text-bearing diagnostic reports.
- This branch implements the first context256 controlled study, not a full architecture capacity increase.
