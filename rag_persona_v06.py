"""v0.6 response planning over v0.5 contextual retrieval.

No audit, approval, or verifier. Extractive by default and non-generative:
the answer is always a contiguous excerpt of an actual retrieved passage.
"""
from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from rag_persona import DEFAULT_PERSONA, GPUBackend
from rag_persona_v03 import LineRAG
from rag_persona_v04 import intent_of
from rag_persona_v05 import SemanticPersona, ASSOCIATIONS

SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])")


@dataclass(frozen=True)
class ResponsePlan:
    strategy: str
    topic: str | None
    intent: str
    source: str | None
    excerpt: str | None


def select_excerpt(passage: str, focus: str | None, parent: str | None,
                   max_chars: int = 180) -> str:
    """Select an exact substring. Never synthesize a sentence or a fact."""
    if not passage:
        return ""
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(passage) if s.strip()]
    if not sentences:
        return ""
    anchors = ASSOCIATIONS.get(parent or "", (parent,) if parent else ())
    def key(sentence: str):
        return (4 * bool(focus and focus in sentence)
                + 2 * bool(parent and parent != focus and parent in sentence)
                + sum(bool(w in sentence) for w in anchors if w and w != focus))
    best = max(enumerate(sentences), key=lambda pair: (key(pair[1]), -pair[0]))[1]
    # Return the complete sentence rather than a truncated nonsensical claim.
    # max_chars is only a preference; truncation would break source fidelity.
    if len(best) > max_chars:
        shorter = [s for s in sentences if len(s) <= max_chars and (not focus or focus in s)]
        if shorter:
            best = max(enumerate(shorter), key=lambda pair: (key(pair[1]), -pair[0]))[1]
    return best


class PlannedPersona(SemanticPersona):
    def __init__(self, rag, persona, mode="extractive", backend=None):
        super().__init__(rag, persona, mode, backend)
        self.last_plan = ResponsePlan("NONE", None, "TOPIC", None, None)

    def clear(self):
        super().clear()
        self.last_plan = ResponsePlan("NONE", None, "TOPIC", None, None)

    def ask(self, question: str) -> str:
        answer = super().ask(question)
        if not question.strip():
            return answer
        intent = intent_of(question)
        topic = self.state.focus_topic
        if self.mode == "llm":
            self.last_plan = ResponsePlan("LLM_COMPARISON", topic, intent, None, None)
            return answer
        if self.last_sources:
            match = re.search(r"「([\s\S]*)」\s*$", answer)
            if match:
                original = match.group(1)
                excerpt = select_excerpt(original, topic, self.state.main_topic)
                self.last_plan = ResponsePlan("EXCERPT", topic, intent, self.last_sources[0], excerpt)
                return "資料の記述では、\n「" + excerpt + "」"
        strategy = "PERSONA" if intent in ("IDENTITY", "GREETING", "PREFERENCE") else "NO_MATCH"
        self.last_plan = ResponsePlan(strategy, topic, intent, None, None)
        return answer


def main():
    p = argparse.ArgumentParser(description="LLM_RAG v0.6 grounded response planner")
    p.add_argument("--docs", type=Path, default=Path("data/knowledge"))
    p.add_argument("--persona", type=Path, default=Path("persona.json"))
    p.add_argument("--mode", choices=("extractive", "llm"), default="extractive")
    p.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    p.add_argument("--checkpoint", type=Path, default=Path("model/model-rag-nagato-v0.2.pt"))
    p.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    persona = dict(DEFAULT_PERSONA)
    if args.persona.is_file():
        persona.update(json.loads(args.persona.read_text(encoding="utf-8")))
    backend = GPUBackend(args.gpu_dir, args.checkpoint, args.tokenizer, args.device) if args.mode == "llm" else None
    chat = PlannedPersona(LineRAG(args.docs), persona, args.mode, backend)
    print(f"LLM_RAG v0.6 | mode={args.mode} | persona={persona['name']} | passages={len(chat.rag.passages)}")
    print("Commands: /status, /reload, /clear, /exit")
    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("/exit", "exit", "quit"):
            break
        if q == "/clear":
            chat.clear()
            print("Conversation cleared.")
        elif q == "/reload":
            chat.rag.refresh()
            chat.clear()
            print(f"RAG reloaded: {len(chat.rag.passages)} passages")
        elif q == "/status":
            s = chat.state
            print(f"main_topic={s.main_topic or '-'} focus_topic={s.focus_topic or '-'} "
                  f"intent={s.intent} used_passages={len(s.seen)} turns={len(s.turns)} "
                  f"response_strategy={chat.last_plan.strategy}")
        elif q:
            print(f"{persona['name']}> {chat.ask(q)}")
            if chat.last_sources:
                print("RAG sources:", ", ".join(chat.last_sources))


if __name__ == "__main__":
    main()
