# LLM_RAG v0.8.8 — Counterfactual Context-Swap Evaluation

## Purpose
v0.8.7 revealed a cross-over: the context-trained model performs better on context prompts (NLL 2.37432 vs 2.50565), but the plain-trained model performs better on plain prompts (2.36493 vs 2.57354). This alone cannot distinguish semantic use of prior dialogue from familiarity with a prompt format.

v0.8.8 therefore keeps the question and answer unchanged and changes ONLY earlier dialogue context on held-out test examples. The model checkpoint is frozen.

## Controls
- Original authentic context from local held-out v0.8.7 test pairs.
- Random unrelated context from another dialogue file; exclude same question or identical context.
- Length-matched context from another dialogue file; prefer exact matching token length, with default tolerance 6.
- Candidate context must fit the 64-token model and tokenizer constraints without truncation.
- Compare per-example mean answer-token NLL: margin = NLL(answer | mismatched history, question) - NLL(answer | original history, question).
- Positive margin means the original context improves answer-token prediction on that example. This is not a human-rated semantic understanding or factual correctness score.
- Data source remains separate by dialogue file, but participants/topics/near-duplicates may overlap.

## Windows PowerShell
    cd C:\Users\inoma\git\misc\LLM_RAG
    git fetch origin
    git switch feature/context-swap-evaluation-v0.8.8
    python -m unittest discover -s tests -v

Run on 200 held-out contextual examples:
    python evaluate_context_swap_v088.py --checkpoint model/model-rag-paired-context-v0.8.7.pt --test-data data/dialogue/real_persona_context_v087/paired_context_test_v087.jsonl --sample 200 --device auto

Result: results/context_swap_v088.json. Does not include raw dialogue text. The selected donor's identity is represented by a short hash; no source text is logged by default.

To inspect a small text-bearing report only locally (do not commit to GitHub):
    python evaluate_context_swap_v088.py --sample 20 --include-text --output results/context_swap_text_v088.json --device auto

## Interpretation
Compare random and length-matched controls separately. Length matching reduces the confound from extra prompt tokens but cannot control semantic plausibility. Count exclusions due to lack of suitable donors, and do not compare different sets of usable examples as if they were identical.

Main follow-up: evaluate the same held-out examples with the plain-trained checkpoint as a negative-control model, and consider matched topic/different meaning interventions. The eventual 256-token model must be built with a compatible checkpoint, not enabled by passing a larger dataset length to this script.

## Protection
Read-only source files and checkpoints. Pre-existing results are not overwritten without --overwrite. Text-bearing reports require a distinctive *_text_v088.json file name. Test outputs and local private data remain local.
