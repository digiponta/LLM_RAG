"""LLM_RAG v0.3 — audit-free extractive persona with optional LLM_GPU mode.

Default mode quotes retrieved text verbatim; it does NOT pretend that the
64-context-token LLM_GPU can reliably synthesize Japanese RAG responses.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from rag_persona import DEFAULT_PERSONA, GPUBackend, LocalRAG, Passage, PersonaChat, SUPPORTED

SUFFIXES = (
    "についてどう考えますか", "についてどう考える", "について教えてください",
    "について教えて", "についてどう思う", "とは何ですか", "とはなんですか",
    "について", "とは", "は何ですか", "を教えてください", "を教えて",
    "をどう思う", "どう思いますか", "どう思う", "ですか", "ますか",
)


def extract_topic(question: str) -> str:
    q = re.sub(r"\s+", "", question).strip("。！？?!、 ")
    for suffix in SUFFIXES:
        if q.endswith(suffix):
            q = q[:-len(suffix)]
            break
    return q.strip("。！？?!、 ")


class LineRAG(LocalRAG):
    """Treat every newline-separated Nagato post as one retrievable record."""

    def refresh(self) -> None:
        self.passages = []
        if not self.docs.exists():
            return
        for path in sorted(self.docs.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            for raw in self._read(path).splitlines():
                sentence = raw.strip()
                if not sentence or sentence.startswith("@NagatFakeBot Tweet Messages"):
                    continue
                self.passages.append(Passage(str(path.relative_to(self.docs)), sentence))


class ExtractivePersona:
    def __init__(self, rag: LineRAG, persona: dict, backend=None, mode="extractive"):
        if mode not in ("extractive", "llm"):
            raise ValueError("invalid mode")
        if mode == "llm" and backend is None:
            raise ValueError("LLM backend is required")
        self.rag, self.persona, self.backend, self.mode = rag, persona, backend, mode
        self.llm_chat = PersonaChat(rag, backend, persona) if backend is not None else None
        self.history: list[tuple[str, str]] = []
        self.last_sources: list[str] = []

    def ask(self, question: str) -> str:
        question = question.strip()
        self.last_sources = []
        if not question:
            return ""
        if self.mode == "llm":
            answer = self.llm_chat.ask(question)
            self.last_sources = list(self.llm_chat.last_sources)
        elif re.fullmatch(r"(こんにちは|こんばんは|おはよう)(。|！|!)*", question):
            answer = question.rstrip("。！!")
        elif "名前" in question and ("あなた" in question or "君" in question or "お名前" in question):
            answer = f"{self.persona['name']}。"
        elif ("読書" in question or "本" in question) and ("好き" in question or "読む" in question):
            answer = "好き。静かな場所で本を読むのがいい。"
        else:
            topic = extract_topic(question)
            if len(topic) < 2:
                answer = "何について知りたい？"
            else:
                # For short topics require contiguous literal occurrence.
                # Long topics permit substantial shared character n-grams.
                candidates = self.rag.search(topic, 8)
                candidates = [p for p in candidates
                              if (topic in p.text if len(topic) <= 5 else p.score >= 0.3)]
                if candidates:
                    best = candidates[0]
                    self.last_sources = [best.source]
                    answer = f"資料では、こう述べられている。\n「{best.text}」"
                else:
                    answer = "その話題について、手元の資料から適切な文章を見つけられなかった。"
        self.history.append((question, answer))
        return answer


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM_RAG v0.3 artificial persona")
    ap.add_argument("--docs", type=Path, default=Path("data/knowledge"))
    ap.add_argument("--persona", type=Path, default=Path("persona.json"))
    ap.add_argument("--mode", choices=("extractive", "llm"), default="extractive")
    ap.add_argument("--gpu-dir", type=Path, default=Path("../LLM_GPU"))
    ap.add_argument("--checkpoint", type=Path, default=Path("model/model-rag-nagato-v0.2.pt"))
    ap.add_argument("--tokenizer", type=Path, default=Path("../LLM_GPU/model/tokenizer.json"))
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args()
    persona = dict(DEFAULT_PERSONA)
    if args.persona.exists():
        persona.update(json.loads(args.persona.read_text(encoding="utf-8")))
    backend = GPUBackend(args.gpu_dir, args.checkpoint, args.tokenizer, args.device) if args.mode == "llm" else None
    chat = ExtractivePersona(LineRAG(args.docs), persona, backend, args.mode)
    print(f"LLM_RAG v0.3 | mode={args.mode} | persona={persona['name']} | passages={len(chat.rag.passages)}")
    print("Commands: /reload, /clear, /exit")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("/exit", "exit", "quit"):
            break
        if question == "/reload":
            chat.rag.refresh()
            print(f"RAG: {len(chat.rag.passages)} passages")
            continue
        if question == "/clear":
            chat.history.clear()
            if chat.llm_chat:
                chat.llm_chat.history.clear()
            print("Conversation cleared.")
            continue
        if question:
            print(f"{persona['name']}> {chat.ask(question)}")
            if chat.last_sources:
                print("RAG sources:", ", ".join(chat.last_sources))


if __name__ == "__main__":
    main()
