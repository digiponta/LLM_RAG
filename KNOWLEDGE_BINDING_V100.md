# LLM_RAG v0.10.0 — Knowledge Binding & Evidence-Conditioned SFT

## Design

v0.9.9 jointly trained Real Persona Chat and Nagato, improving held-out Nagato NLL 4.66322 to 2.90268 and Persona NLL 2.34895 to 2.26831, but free generation continued to reply 'そうですね' to topic questions. v0.10.0 targets the missing link: user question + selected evidence -> attributed answer. It uses **exactly the same tokenizer, RoPE Medium network, compact prompt, answer-only loss and training loop** as v0.9.9, continuing from the user's already trained v0.9.9 checkpoint.

Important distinction: This is **oracle evidence conditioning** for a controlled research experiment, **not automatic knowledge selection**. The local dataset builder places a short verbatim Nagato source excerpt inside the existing context field (formatted as '資料:...'). For every held-out question, a paired no-evidence probe preserves the same reference answer but removes the source text. A difference in NLL between the two conditions demonstrates dependence on supplied text, NOT free-answer knowledge or independently verified truth.

Nagato statements can contain unverified speculation and fiction: outputs quote the source and are not scientific assertions. Existing Real Persona Chat conversations and all generated local files remain on the user's machine.

## 1. Update + run tests

    cd C:\Users\inoma\git\misc\LLM_RAG
    git fetch origin
    git switch feature/knowledge-binding-v0.10.0
    python -m unittest discover -s tests -v

Expected 266 + 17 = **283 tests**. Results must be verified on the user's Windows machine.

## 2. Prepare v0.10.0 dataset (no private uploads)

    python prepare_knowledge_binding_v100.py `
      --output-dir data/dialogue/binding_v100 `
      --nagato-fraction 0.05 `
      --excerpt-chars 56 `
      --context-length 256

The source Real Persona train/validation inputs default to v0.9.9 unified JSONLs but their previously oversampled Nagato rows are removed. The Nagato inputs are the ORIGINAL unweighted v0.9.9 Nagato train/validation/test JSONLs. Only source-attributed rows are used for constructing conditioned direct and follow-up tasks. For each source excerpt, the builder emits 2 variants: (a) direct question about an extracted topic with verbatim evidence, (b) a follow-up asking for the passage with topic, previous question and same evidence. Both are filtered by the actual existing SFT encoder (maximum 256 tokens). Source lines stay disjoint across train/validation/test. An ablation file strips the evidence while holding each answer fixed.

Files created:

    data/dialogue/binding_v100/binding_train_v100.jsonl
    data/dialogue/binding_v100/binding_validation_v100.jsonl
    data/dialogue/binding_v100/binding_test_v100.jsonl
    data/dialogue/binding_v100/binding_test_no_evidence_v100.jsonl
    data/dialogue/binding_v100/binding_report_v100.json

The report prints source counts but no private text. Do not commit these files. If too many follow-up rows exceed context budget, reduce --excerpt-chars to 40 and choose a new output directory. Do not overwrite existing data by default.

## 3. Continue SFT on the SAME unified LLM

    python train_real_dialogue_v082.py `
      --checkpoint model/model-rag-unified-nagato-v0.9.9.pt `
      --source data/dialogue/binding_v100/binding_train_v100.jsonl `
      --validation data/dialogue/binding_v100/binding_validation_v100.jsonl `
      --max-train 0 --max-validation 0 `
      --epochs 1 --batch-size 4 --learning-rate 0.00002 `
      --rope --device auto `
      --output model/model-rag-binding-v0.10.0.pt `
      --report results/binding_train_v100.json

## 4. Compare held-out conditioned vs unconditioned

Bound test (oracle evidence supplied):

    python evaluate_real_dialogue_v082.py `
      --baseline model/model-rag-unified-nagato-v0.9.9.pt --baseline-rope `
      --tuned model/model-rag-binding-v0.10.0.pt --tuned-rope `
      --test-data data/dialogue/binding_v100/binding_test_v100.jsonl `
      --sample 500 --generations 20 --batch-size 4 `
      --output results/binding_test_v100.json --device auto

Identical questions and answers, WITHOUT evidence:

    python evaluate_real_dialogue_v082.py `
      --baseline model/model-rag-unified-nagato-v0.9.9.pt --baseline-rope `
      --tuned model/model-rag-binding-v0.10.0.pt --tuned-rope `
      --test-data data/dialogue/binding_v100/binding_test_no_evidence_v100.jsonl `
      --sample 500 --generations 20 --batch-size 4 `
      --output results/binding_no_evidence_test_v100.json --device auto

Also rerun the untouched Real Persona Chat test (1,000 questions), to detect loss of dialogue ability. Do not assume evidence-conditioned NLL improvements mean free-form facts can be generated.

## 5. Inspect free conversation

    python chat.py `
      --checkpoint model/model-rag-binding-v0.10.0.pt `
      --rag --docs data/knowledge `
      --evidence-chars 56 `
      --temperature 0 --repetition-penalty 1.0 `
      --device auto

Chat retrieval uses character n-gram matching, not oracle evidence. It can retrieve the wrong excerpt and may truncate evidence to fit context. The exact training input should be reconstructed from the test file locally to distinguish actual retrieval failure from failure to use evidence. Check /sources. For a clean A/B, also run chat without --rag.

## Limitations

- Same topic may appear across many Nagato lines; an ungrounded generic question has no unique reference answer. The evidence-conditioned task intentionally resolves that ambiguity with an explicit excerpt.
- Quoting source text as the answer avoids inventing facts but is an easier copy task than explanation, paraphrase or summarization. The follow-up output repeats the passage instead of generating NEW details. It therefore tests conditional grounding, not multi-turn reasoning.
- Test splits exclude exact original Nagato lines used in the training source, but not all lexical/semantic overlap. This is development evaluation, not independent third-party holdout.
- Existing 5,117-character tokenizer and context=256 severely limit both evidence and response length. Metadata and actual source text remain available locally for later richer evaluation.
