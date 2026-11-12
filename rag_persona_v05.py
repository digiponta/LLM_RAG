"""LLM_RAG v0.5: context-aware extractive retrieval (no audit/approval)."""
from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from rag_persona import DEFAULT_PERSONA, GPUBackend, Passage, terms
from rag_persona_v03 import LineRAG, ExtractivePersona
from rag_persona_v04 import intent_of, explicit_topic

ASSOCIATIONS = {
    "宇宙": ("宇宙", "時空", "観測", "粒子", "量子", "相対性", "重力", "物理", "光", "空間"),
    "量子力学": ("量子", "粒子", "波動", "状態", "観測", "エネルギ", "物理"),
}
COMPUTING = ("Win7", "Windows", "ブート", "CPU", "パソコン", "OS")
FOLLOWON_SUFFIXES = ("については", "に関しては", "の話は")

@dataclass
class SemanticState:
    main_topic: str | None = None
    focus_topic: str | None = None
    intent: str = "TOPIC"
    seen: set[str] = field(default_factory=set)
    turns: list[tuple[str, str]] = field(default_factory=list)

    def clear(self):
        self.main_topic = self.focus_topic = None
        self.intent = "TOPIC"
        self.seen.clear()
        self.turns.clear()

def related_followon(q: str, main: str | None) -> bool:
    q = re.sub(r"\s+", "", q).rstrip("。？！?! ")
    return bool(main and any(q.endswith(x) for x in FOLLOWON_SUFFIXES))

def semantic_rank(passages: list[Passage], focus: str, parent: str | None,
                  intent: str, seen: set[str], limit: int = 5) -> list[Passage]:
    if not focus or limit < 1:
        return []
    focusgrams = terms(focus)
    linked = bool(parent and parent != focus)
    anchors = ASSOCIATIONS.get(parent or "", (parent,) if parent else ())
    ranked = []
    for p in passages:
        hit = focus in p.text
        similarity = len(focusgrams & terms(p.text)) / max(1, len(focusgrams))
        if not hit and (len(focus) <= 5 or similarity < 0.5):
            continue
        score = (4.0 if hit else 0.0) + similarity
        if linked:
            evidence = sum(1 for word in anchors if word != focus and word in p.text)
            if evidence == 0:
                continue
            score += min(2.5, evidence * 0.8)
            if parent in p.text:
                score += 2.0
            if parent == "宇宙" and focus == "時間" and any(w in p.text for w in COMPUTING):
                score -= 6.0
        if intent == "DEFINITION" and any(x in p.text for x in (focus+"は", focus+"とは")):
            score += 0.5
        if intent == "OPINION" and any(x in p.text for x in ("思う", "考え", "可能性", "興味深い")):
            score += 0.4
        if p.text not in seen and score > 0:
            ranked.append((score, p))
    ranked.sort(key=lambda x: (-x[0], x[1].source, x[1].text))
    selected = []
    for score, p in ranked:
        if any(len(terms(p.text) & terms(s.text)) / max(1, min(len(terms(p.text)), len(terms(s.text)))) > 0.85 for s in selected):
            continue
        selected.append(Passage(p.source, p.text, score))
        if len(selected) == limit:
            break
    return selected

class SemanticPersona:
    def __init__(self, rag: LineRAG, persona: dict, mode: str = "extractive", backend=None):
        if mode not in ("extractive", "llm"):
            raise ValueError("Invalid mode")
        self.rag, self.persona, self.mode = rag, persona, mode
        self.state = SemanticState()
        self.last_sources = []
        self.comparison = ExtractivePersona(rag, persona, backend, "llm") if mode == "llm" else None

    def clear(self):
        self.state.clear()
        self.last_sources = []
        if self.comparison:
            self.comparison.history.clear()
            self.comparison.llm_chat.history.clear()

    def ask(self, question: str) -> str:
        q = question.strip()
        self.last_sources = []
        if not q:
            return ""
        if self.comparison:
            answer = self.comparison.ask(q)
            self.last_sources = list(self.comparison.last_sources)
            return answer
        intent = intent_of(q)
        if intent == "GREETING":
            answer = q.rstrip("。！!")
        elif intent == "IDENTITY":
            answer = self.persona["name"] + "。"
        elif intent == "PREFERENCE":
            answer = "好き。静かな場所で本を読むのがいい。"
        else:
            topic = explicit_topic(q, intent)
            if topic and topic != self.state.focus_topic:
                if related_followon(q, self.state.main_topic):
                    self.state.focus_topic = topic
                else:
                    self.state.main_topic = topic
                    self.state.focus_topic = topic
                self.state.seen.clear()
            if self.state.focus_topic is None:
                answer = "どの話題について知りたい？"
            else:
                actual_intent = self.state.intent if intent == "FOLLOWUP" else intent
                matches = semantic_rank(self.rag.passages, self.state.focus_topic,
                                        self.state.main_topic, actual_intent, self.state.seen)
                if matches:
                    chosen = matches[0]
                    self.state.seen.add(chosen.text)
                    self.last_sources = [chosen.source]
                    prefix = "関連する別の記述がある。" if intent == "FOLLOWUP" else "資料には、こう記されている。"
                    answer = prefix + "\n「" + chosen.text + "」"
                else:
                    answer = "「" + self.state.focus_topic + "」について、現在の文脈に合う別の記述が見つからなかった。"
                if intent != "FOLLOWUP":
                    self.state.intent = intent
        self.state.turns.append((q, answer))
        return answer

def main():
    p = argparse.ArgumentParser(description="LLM_RAG v0.5 contextual semantic RAG")
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
    rag = LineRAG(args.docs)
    chat = SemanticPersona(rag, persona, args.mode, backend)
    print(f"LLM_RAG v0.5 | mode={args.mode} | persona={persona['name']} | passages={len(rag.passages)}")
    print("Commands: /status, /reload, /clear, /exit")
    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("exit", "quit", "/exit"):
            break
        if q == "/clear":
            chat.clear()
            print("Conversation cleared.")
        elif q == "/reload":
            rag.refresh()
            chat.clear()
            print(f"RAG reloaded: {len(rag.passages)} passages")
        elif q == "/status":
            s = chat.state
            print(f"main_topic={s.main_topic or '-'} focus_topic={s.focus_topic or '-'} intent={s.intent} used_passages={len(s.seen)} turns={len(s.turns)}")
        elif q:
            print(f"{persona['name']}> {chat.ask(q)}")
            if chat.last_sources:
                print("RAG sources:", ", ".join(chat.last_sources))

if __name__ == "__main__":
    main()
