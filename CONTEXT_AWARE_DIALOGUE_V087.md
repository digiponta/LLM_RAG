# LLM_RAG v0.8.7 — Context-Aware Paired Dialogue

## Goal
Test whether real preceding dialogue turns improve question-conditioned response generation. Preserve the v0.8.4 model and the original dialogue-file train/validation/test split.

## Data design
- Read local original Real Persona Chat JSON using each v0.8.1 conversation_id and source index.
- Validate that the source question and answer still match the v0.8.1 records.
- Extract one or two immediately preceding authentic turns in chronological order, where the closest earlier turn is from the responding speaker.
- Never synthesize history, truncate responses, or merge across broken turns.
- Create paired plain and contextual JSONL files: identical IDs, questions, answers and file splits; only the context field differs.
- Test exact sequence encoding with the original tokenizer against context length 64. Keep only records fitting both variants.
- New data contain private chat text and must remain local. The summary JSON is aggregate-only.

## Step 1: Checkout and test
PowerShell:
    cd C:\Users\inoma\git\misc\LLM_RAG
    git fetch origin
    git switch feature/context-aware-dialogue-v0.8.7
    python -m unittest discover -s tests -v

## Step 2: Create paired datasets
    python prepare_context_dialogue_v087.py --context-length 64 --turns 2
Result: data/dialogue/real_persona_context_v087/ with six paired train/validation/test files and context_report_v087.json. Review the accepted counts first: 64 tokens severely restrict how much history can be retained. DO NOT run training if the paired cohort is tiny.

## Step 3: Fine-tune both variants from the same frozen v0.8.4 checkpoint
    python train_real_dialogue_v082.py --checkpoint model/model-rag-real-dialogue-full-v0.8.4.pt --source data/dialogue/real_persona_context_v087/paired_plain_train_v087.jsonl --validation data/dialogue/real_persona_context_v087/paired_plain_validation_v087.jsonl --max-train 0 --max-validation 0 --epochs 2 --learning-rate 0.00005 --output model/model-rag-paired-plain-v0.8.7.pt --report results/context_plain_train_v087.json --device auto
    python train_real_dialogue_v082.py --checkpoint model/model-rag-real-dialogue-full-v0.8.4.pt --source data/dialogue/real_persona_context_v087/paired_context_train_v087.jsonl --validation data/dialogue/real_persona_context_v087/paired_context_validation_v087.jsonl --max-train 0 --max-validation 0 --epochs 2 --learning-rate 0.00005 --output model/model-rag-paired-context-v0.8.7.pt --report results/context_conditioned_train_v087.json --device auto

Both models see exactly the same question/answer pairs. Training is continued SFT, not from-scratch retraining. Checkpoints and original data are not overwritten.

## Step 4: Evaluate on identical held-out prompts
    python evaluate_real_dialogue_v082.py --baseline model/model-rag-paired-plain-v0.8.7.pt --tuned model/model-rag-paired-context-v0.8.7.pt --test-data data/dialogue/real_persona_context_v087/paired_context_test_v087.jsonl --sample 500 --generations 20 --output results/context_matched_test_v087.json --device auto
Repeat with paired_plain_test_v087.jsonl and a distinct output path to distinguish context prompt effects. NLL alone does not establish response relevance or fluency; inspect generation samples separately.

## Constraints
- Existing v0.8.1 split is by dialogue file, not speaker or near-duplicate.
- The paired dataset is a selected cohort; its NLL is not directly comparable to the full v0.8.4 test score.
- Passing context-length 256 to the converter does not create a 256-token model. That needs a separately constructed and evaluated LLM_GPU checkpoint.
- Training and evaluation have not yet been run on the user's Windows GPU.
