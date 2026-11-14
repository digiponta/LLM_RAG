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
    if "prompt_override" in row:
        raise ValueError("Refuse mixed structured prompt_override: no deterministic injection")
    clone = dict(row)
    context = row.get("context") or ""
    tag = TAG_BY_TASK[mode]
    if context.startswith("[TASK:"):
        raise ValueError("Row already contains a task marker")
    clone["context"] = tag + ("\n" + context if context else "")
    clone["task_marker"] = tag
    return clone
