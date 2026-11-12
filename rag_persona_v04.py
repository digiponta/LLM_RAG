"""LLM_RAG v0.4: context-aware, audit-free extractive artificial persona.

Works without GPU in the default extractive mode. Original source text is
quoted rather than rewritten or presented as verified scientific fact.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rag_persona import DEFAULT_PERSONA, GPUBackend, Passage, terms
from rag_persona_v03 import LineRAG, ExtractivePersona, extract_topic


FOLLOWUPS = ("もう少し詳しく", "もっと詳しく", "詳しく", "詳しく教えて",
             "続きを", "続きは", "それについて", "その話", "もう少し", "なぜ", "どうして")
TOPIC_MARKERS = ("について", "とは", "に関して", "のこと", "はどう")
STRIP_END = ("どう考えますか", "どう考える", "どう思いますか", "どう思う",
             "教えてください", "教えて", "何ですか", "なんですか", "は？", "は",
             "ですか", "について", "とは")
SHORT_GENERIC = {"それ", "その", "これ", "あれ", "そう", "詳しく", "続き", "もっと"}
PARTICLES = ("について", "に関して", "は", "を", "が", "の")
SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])")


def clean(text: str) -> str:
    return re.sub(r"\s+", "", text).strip("。！？?!、 ")


def intent_of(question: str) -> str:
    q = clean(question)
    if any(q.startswith(x) for x in FOLLOWUPS) or q in SHORT_GENERIC:
        return "FOLLOWUP"
    if "名前" in q and ("あなた" in q or "君" in q or "お名前" in q):
        return "IDENTITY"
    if re.fullmatch(r"(こんにちは|こんばんは|おはよう)(。|！|!)*", question.strip()):
        return "GREETING"
    if ("読書" in q or "本" in q) and ("好き" in q or "読む" in q):
        return "PREFERENCE"
    if any(x in q for x in ("どう考え", "どう思", "意見")):
        return "OPINION"
    if any(x in q for x in ("とは", "何ですか", "なんですか")):
        return "DEFINITION"
    return "TOPIC"


def explicit_topic(question: str, intent: str) -> str | None:
    if intent in ("FOLLOWUP", "GREETING", "IDENTITY", "PREFERENCE"):
        return None
    q = clean(question)
    for marker in TOPIC_MARKERS:
        if marker in q:
            prefix = q.split(marker, 1)[0]
            if prefix and prefix not in SHORT_GENERIC:
                return prefix
    for suffix in STRIP_END:
        if q.endswith(suffix):
            q = q[:-len(suffix)]
            break
    for suffix in PARTICLES:
        if q.endswith(suffix):
            q = q[:-len(suffix)]
    return q if len(q) >= 2 and q not in SHORT_GENERIC else None


def rank_passages(rag: LineRAG, topic: str, intent: str,
                  prior_texts: set[str], k: int = 12) -> list[Passage]:
    """Topic coverage + explicit substring + intent preference + novelty.

    A short explicit topic must occur literally; this avoids accidental
    n-gram matches when the question is short (e.g. 宇宙).
    """
    if not topic:
        return []
    qgrams = terms(topic)
    found = []
    for p in rag.passages:
        body = p.text
        topic_match = topic in body
        overlap = len(qgrams & terms(body)) / max(1, len(qgrams))
        if len(topic) <= 5 and not topic_match:
            continue
        if not topic_match and overlap < 0.42:
            continue
        score = (3.0 if topic_match else 0.0) + overlap
        if intent == "DEFINITION":
            if any(x in body for x in (f"{topic}とは", f"{topic}は", f"{topic}という", "定義")):
                score += 1.0
        elif intent == "OPINION":
            if any(x in body for x in ("思う", "考え", "可能性", "興味深い", "かもしれない")):
                score += 0.5
        if body in prior_texts:
            score -= 10.0
        found.append((score, p))
    found.sort(key=lambda item: (-item[0], item[1].source, item[1].text))
    return [Passage(p.source, p.text, score) for score, p in found[:k] if score > 0]


@dataclass
class DialogueState:
    topic: str | None = None
    intent: str = "TOPIC"
    seen: set[str] = field(default_factory=set)
    turns: list[tuple[str, str]] = field(default_factory=list)

    def clear(self):
        self.topic = None
        self.intent = "TOPIC"
        self.seen.clear()
        self.turns.clear()


class ContextPersona:
    def __init__(self, rag: LineRAG, persona: dict, mode: str = "extractive", backend=None):
        if mode not in ("extractive", "llm"):
            raise ValueError("mode must be extractive or llm")
        self.mode, self.persona, self.rag = mode, persona, rag
        self.state = DialogueState()
        self.last_sources: list[str] = []
        self.old_llm = ExtractivePersona(rag, persona, backend, mode="llm") if mode == "llm" else None

    def clear(self):
        self.state.clear()
        self.last_sources.clear()
        if self.old_llm:
            self.old_llm.history.clear()
            self.old_llm.llm_chat.history.clear()

    def ask(self, question: str) -> str:
        question = question.strip()
        self.last_sources = []
        if not question:
            return ""
        if self.mode == "llm":
            answer = self.old_llm.ask(question)
            self.last_sources = list(self.old_llm.last_sources)
            return answer
        intent = intent_of(question)
        if intent == "GREETING":
            answer = clean(question)
        elif intent == "IDENTITY":
            answer = f"{self.persona['name']}。"
        elif intent == "PREFERENCE":
            answer = "好き。静かな場所で本を読むのがいい。"
        else:
            new_topic = explicit_topic(question, intent)
            if new_topic is not None and new_topic != self.state.topic:
                self.state.topic = new_topic
                self.state.seen.clear()
            if self.state.topic is None:
                answer = "どの話題について知りたい？"
            else:
                effective_intent = self.state.intent if intent == "FOLLOWUP" else intent
                matches = rank_passages(self.rag, self.state.topic, effective_intent, self.state.seen)
                if matches:
                    selected = matches[0]
                    self.state.seen.add(selected.text)
                    self.last_sources = [selected.source]
                    lead = "関連する別の記述がある。" if intent == "FOLLOWUP" else "資料には、こう記されている。"
                    answer = f"{lead}\n「{selected.text}」"
                else:
                    answer = (f"「{self.state.topic}」について、これ以上関連する記述は見つからなかった。"
                              if self.state.seen else
                              f"「{self.state.topic}」について、資料に該当する記述が見つからなかった。")
                if intent != "FOLLOWUP":
                    self.state.intent = intent
        self.state.turns.append((question, answer))
        return answer


def main():
    parser = argparse.ArgumentParser(description="LLM_RAG v0.4 context-aware persona")
    parser.add_argument("--docs", type=Path, default=Path("data/knowledge"))
    parser.add_argument("--persona", type=Path, default=Path("persona.json"))
    parser.add_argument("--mode", choices=("extractive", "llm"), default="extractive")
    parser.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    parser.add_argument("--checkpoint", type=Path, default=Path("model/model-rag-nagato-v0.2.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    persona = dict(DEFAULT_PERSONA)
    if args.persona.exists():
        persona.update(json.loads(args.persona.read_text(encoding="utf-8")))
    rag = LineRAG(args.docs)
    backend = (GPUBackend(args.gpu_dir, args.checkpoint, args.tokenizer, args.device)
               if args.mode == "llm" else None)
    chat = ContextPersona(rag, persona, mode=args.mode, backend=backend)
    print(f"LLM_RAG v0.4 | mode={args.mode} | persona={persona['name']} | passages={len(rag.passages)}")
    print("Commands: /status, /reload, /clear, /exit")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("exit", "quit", "/exit"):
            break
        if question == "/clear":
            chat.clear()
            print("Conversation cleared.")
            continue
        if question == "/reload":
            rag.refresh()
            chat.clear()
            print(f"RAG reloaded: {len(rag.passages)} passages; context cleared")
            continue
        if question == "/status":
            print(f"topic={chat.state.topic or '-'} intent={chat.state.intent} "
                  f"used_passages={len(chat.state.seen)} turns={len(chat.state.turns)}")
            continue
        if question:
            print(f"{persona['name']}> {chat.ask(question)}")
            if chat.last_sources:
                print("RAG sources:", ", ".join(chat.last_sources))


if __name__ == "__main__":
    main()
