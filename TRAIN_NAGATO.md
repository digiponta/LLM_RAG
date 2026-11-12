# Nagato continued training (LLM_RAG v0.2)

This script changes **model weights**. It fine-tunes the existing LLM_GPU
checkpoint on the authored text in \`data/knowledge/data-nagato.txt\`, using
next-token cross entropy. It does not generate question/answer labels and is
not dialogue-specific instruction tuning. The corpus includes speculation,
creative fiction and opinions, not necessarily verified facts.

## Windows PowerShell

From the LLM_RAG repo and the feature branch:

```powershell
python -m unittest discover -s tests -v

# Adjust the source checkpoint if your local LLM_GPU v0.4 uses another name.
python train_nagato.py  --gpu-dir ..\LLM_GPU 
    --checkpoint ..\LLM_GPU\model\model-gpu-v0.3.pt 
    --tokenizer ..\LLM_GPU\model\tokenizer.json 
    --data data\knowledge\data-nagato.txt 
    --output model\model-rag-nagato-v0.2.pt 
    --epochs 5 --batch-size 32 --device auto
```

Output: `model/model-rag-nagato-v0.2.pt` and
`results/nagato_finetune_v02.json`.

Start with the fine-tuned checkpoint:

```powershell
python rag_persona.py --gpu-dir ../LLM_GPU  --checkpoint model/model-rag-nagato-v0.2.pt 
    --tokenizer ../LLM_GPU/model/tokenizer.json 
    --docs data/knowledge --device auto
```

**Safety of existing data:** The original checkpoint and tokenizer are never
overwritten. If the destination files exist, the script refuses to overwrite
unless `--overwrite` is supplied.

**Tokenizer:** The existing vocabulary is preserved to maintain compatibility
with the source weights. Fine-tuning aborts if too many characters tokenize as
`<UNK>` (default threshold 3%). If the checkpoint has a different vocabulary
or model layout, choose matching input artifacts; do not blindly resize layers.

**Evaluation:** Held-out lines are never used in parameter updates.
Check both validation loss and real dialogue quality. A falling loss does not
prove artificial persona or RAG QA correctness. With the current 64-token
context, long RAG prompts still truncate. Use separate dialogue SFT and larger
context training for the next phase.
