# v0.10.12 — Frozen Model / Sealed TEST Evaluation Protocol

## Freeze the candidate before any final TEST work

**Frozen candidate**: local \`model/model-rag-position-rehearsal-v0.10.11.pt\`.
**Frozen baseline**: \`model/model-rag-binding-v0.10.0.pt\`.

Development evidence from user-run v0.10.11:
- Position exact **704/1200 = 58.67%** across existing v0.10.3 DEVELOPMENT and swapped counterfactuals; positions 1/5/10 = 61.5/59.5/55.0%.
- Nagato Knowledge NLL **2.6893**, Nagato Persona **3.5206**, Real Persona Chat **2.4269**, oracle Evidence Binding **2.4275**.
- Existing v0.10.7 DEVELOPMENT Retention Gate **DEVELOPMENT_GATE_PASS**; \`accepted_for_deployment=false\`.
- These are *not* independent final-test results.

## Test integrity

Do **not** retrain or choose hyperparameters from final TEST scores. This script has a two-stage interface:

1. **Preflight** (default) hashes the candidate and baseline checkpoints, validates argument paths and leaves every TEST path **unopened and unstat'ed**.
2. **Sealed TEST** (\`--open-final-test\`) reads pre-generated test rows. Prior to model scoring, it checks train/Test source-origin identities, row IDs and exact Q/A duplication and refuses overlaps. Performs answer-token NLL comparison on the exact same held-out rows, plus a tiny deterministic greedy generation diagnostic. Writes a local report using exclusive file creation (never overwrites). Raw replies and raw private Persona text are *not* included in the report.

The original Nagato Test was separated by ORIGINAL line SHA256, rather than chunk. Oracle Evidence Test is derived only from that test partition. Source-level separation is stronger than random rows but not enough to guarantee conceptual independence, deduplicate semantically similar sources, or establish scientific truth. Persona Test must be identified by an explicitly existing untouched file; the script does not infer or create it.

**Important:** The default initial TRAIN audit uses \`data/dialogue/multitask_v107/multitask_train_v107.jsonl\` (the final multi-task TRAIN sampled rows), not every intermediate original source/alternative corpus and every prior checkpoint's full provenance. Record this limitation. Check that all original upstream sources and their split manifests are identical to the ones used for the frozen candidate before claiming a source-independent evaluation. Do not compare later candidates on this TEST after its first opening.

## 1. Git checkout and CPU regression suite

\`\`\`powershell
cd C:\Users\inoma\git\misc\LLM_RAG
git fetch origin
git switch feature/frozen-independent-eval-v0.10.12
python -m unittest discover -s tests -v
\`\`\`

Expected **443 tests** (prior 432 + 11 new). User's local Windows runtime / GPU evaluation have not been executed by the assistant.

## 2. Test preparation guard (does not open TEST files)

\`\`\`powershell
python evaluate_frozen_sealed_test_v1012.py
\`\`\`

Prints \`READY_TO_OPEN_SEALED_TEST\`, candidate/baseline SHA256, intended TEST paths. The default report path is *not written* in preflight. Do not change frozen candidate weights afterward.

## 3. Identify actual Persona TEST path before final opening

The v0.9.9 unified preparation used \`paired_legacy_train_v092.jsonl\` and \`paired_legacy_validation_v092.jsonl\`. This branch does not assume there is a corresponding \`paired_legacy_test_v092.jsonl\`. Inspect locally, without reading TEST rows:

\`\`\`powershell
Get-ChildItem data/dialogue -Recurse -File -Filter "*test*.jsonl" |
    Select-Object -ExpandProperty FullName
\`\`\`

Choose a Persona Test generated from a **distinct source dialogue FILE split** and not touched by any training or model selection. If one is not available, do not pretend the three Nagato/Evidence final TEST tasks cover Real Persona Chat; report that as missing and create a new prospective, independent conversation evaluation in a separate study.

## 4. Optional exploratory Copy evaluation BEFORE final TEST

This uses the OLD DEVELOPMENT data, not independent final test, and must not affect choice of frozen candidate after examining sealed TEST.

\`\`\`powershell
python evaluate_generalized_copy_v103.py \`
    --baseline model/model-rag-copy-general-v0.10.3.pt \`
    --tuned model/model-rag-position-rehearsal-v0.10.11.pt \`
    --dataset data/dialogue/copy_general_v103/copy_general_development_v103.jsonl \`
    --sample 0 --max-new-tokens 32 --device auto \`
    --output results/copy_frozen_development_v1012.json
\`\`\`

The exact 10-character-copy metric is independent of Position extraction only in **task type**, not in held-out source independence; these DEVELOPMENT samples may have been used repeatedly during model development.

## 5. Deliberate once-only sealed TEST (do only if frozen and ready)

Without Persona Test:

\`\`\`powershell
python evaluate_frozen_sealed_test_v1012.py \`
    --open-final-test \`
    --checkpoint model/model-rag-position-rehearsal-v0.10.11.pt \`
    --baseline model/model-rag-binding-v0.10.0.pt \`
    --max-rows 0 --generations 12 \`
    --batch-size 16 --device auto \`
    --report results/frozen_sealed_test_v1012.json
\`\`\`

With verified untouched Persona Test, add \`--persona-test "<absolute-or-relative-existing-path>.jsonl"\` before \`--report\`. The report includes *only* actually evaluated groups; if no Persona TEST is passed, there is NO Real Persona TEST NLL.

After the first final Test use, lock the report, its checkpoint hash, tokenizer and tested file hashes. Do **not** change model weight or thresholds based on final TEST. For future improvements, create a genuinely **new external source-disjoint prospective test**.

## 6. Manual/proxy human-review path (not automated)

Use a separate generated-response review sample, preferably of new conversations **not previously in development prompts**. Score at least persona coherence, topic relevance, unsupported claims, evidence fidelity, long-range followups, and genuine multi-turn turn-taking. Present proxy suggestions for confirmation; only explicit human action grants approval. Do not call automatic exact-string matching an adequate dialogue quality metric.

## What this evaluation does not prove

- Neither Nagato scientific fact truth nor persona authenticity.
- External RAG retrieval performance; the evidence is supplied as **oracle context**.
- Copy/Position on a fresh independent TEST; current position report is DEVELOPMENT only.
- Multi-turn dialogue quality, safety, factual claims outside corpus, or deployment readiness.

**No deployment promotion is automated even after a strong sealed TEST result.**
