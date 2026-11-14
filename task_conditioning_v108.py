"""v0.10.8: explicit task-conditioned input schema for existing RoPE Medium.

Tags are ordinary text, not new tokenizer special tokens. They are added ONLY
to prompt context, never target answers. Same transform must be applied at
training and candidate inference to avoid an input-distribution mismatch.
"""
from __future__ import annotations

from copy import deepcopy

TAG_BY_TASK = {
    "real_persona": "[TASK:CHAT]",
    "nagato_knowledge": "[TASK:KNOWLEDGE]",
    "nagato_persona": "[TASK:PERSONA]",
    "evidence_binding": "[TASK:EVIDENCE]",
    "position": "[TASK:POSITION]",
    "copy": "[TASK:COPY]",
}


def tagged(row, task=None):
    mode = task if task is not None else row.get("multitask")
    if mode not in TAG_BY_TASK:
        raise ValueError(f"Unknown task label: {mode}")
    clone = dict(row)
    tag = TAG_BY_TASK[mode]
    if "prompt_override" in row:
        prompt = row["prompt_override"]
        if not isinstance(prompt, str) or not prompt.endswith("A:"):
            raise ValueError("Invalid prompt_override")
        if prompt.startswith("[TASK:"):
            raise ValueError("Already tagged structured prompt")
        clone["prompt_override"] = tag + "\n" + prompt
    else:
        context = row.get("context") or ""
        if context.startswith("[TASK:"):
            raise ValueError("Row already contains a task marker")
        clone["context"] = tag + ("\n" + context if context else "")
    clone["task_marker"] = tag
    return clone
