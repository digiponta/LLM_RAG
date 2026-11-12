# LLM_RAG v0.8.0 — Generation Collapse Diagnostics

The v0.7 SFT improved held-out-row answer NLL (3.97066 → 3.71378),
but deterministic generation repeated `、` and generic Japanese fragments.
Do **not** infer conversation quality from the lower NLL.

## Diagnostics added

- Stepwise raw EOS probability, EOS rank, top-1 probability, generated token.
- Repetition ratio, longest repeated-token run, repeated bigrams,
  distinct-token ratio, punctuation ratio, emitted length and EOS stopping.
- Identical prompts for source v0.2 and dialogue SFT v0.7 checkpoints.
- Three decode policies: `greedy` (temperature=0, no penalty),
  `penalty` (temperature=0, penalty=1.3), and `sample`
  (temperature=0.75, top-k=20, penalty=1.2).
- Six fixed probes (definition, opinion, preference, identity, follow-up,
  unknown), deliberately distinct from v0.7's topic-only prompt wording.
- JSON report with all token traces and per-model/per-policy aggregates.

## Windows PowerShell

```powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/dialogue-diagnostics-v0.8.0
python -m unittest discover -s tests -v
python diagnose_dialogue_v080.py `
    --gpu-dir ../LLM_GPU `
    --tokenizer ../LLM_GPU/model/tokenizer.json `
    --baseline model/model-rag-nagato-v0.2.pt `
    --tuned model/model-rag-dialogue-v0.7.pt `
    --max-new-tokens 64 `
    --device auto `
    --output results/dialogue_diagnostics_v080.json
```

The prior branch had 51 tests; this branch adds 10 diagnostics tests.
Expected total: 61. If the report already exists, choose a new filename
or explicitly pass `--overwrite`.

## How to interpret

- Very high adjacent-repeat and punctuation ratios under greedy indicate
  degenerate generation.
- Improvement from repetition penalty/sampling indicates decoding has a
  material role; it does not demonstrate correct responses.
- EOS probability/rank staying low through the full 64-token output can
  indicate insufficient learned stopping behavior.
- Repeating similar outputs across diverse questions indicates weak
  conditioning on the prompt.
- Compare full text, not only quantitative proxy metrics.

## Evaluation boundary

These probes are new *prompt formulations*, but the base model has seen
Nagato source corpus data already. This is a diagnostic comparison,
NOT an independent human-reviewed generalization benchmark, and not
a test of factual correctness. No model checkpoint or RAG runtime changes.

## Subsequent planned steps

v0.8.1: new genuinely conversational training examples with separate
held-out conversations; v0.8.2: context length and model-capacity ablations;
v0.8.3: retrain and independently evaluate; v0.8.4: hybrid RAG integration
only if generation becomes sufficiently reliable.
