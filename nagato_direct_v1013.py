"""Conservative proper-name match -> one ORIGINAL sentence from data-nagato.txt.

No LLM generation, summarization, model editing, RAG chunk slicing or hidden
sentence paraphrasing. A source claim is not an independently verified fact.
Japanese proper noun detection without a morphology library is necessarily
heuristic: curated names, optional local name list, quoted titles and some
distinctive katakana/Latin names are supported. Ordinary common nouns are
not intended to trigger. Unknown names can be added to a user-local lexicon.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Safely conservative default list, all still require a literal occurrence
# in the source file AND in the user's question before returning a sentence.
SEED_NAMES = (
    "長門有希", "涼宮ハルヒ", "朝比奈みくる", "イスカンダル",
    "コスモクリーナー", "攻殻機動隊", "ドラえもん", "ガンダム",
    "エヴァンゲリオン", "Windows", "Linux", "Python",
    "RenderMan", "SoftImage", "Maya", "SAP", "SQUID",
)
# Katakana words such as パソコン, ストレス and generic technical terms
# are *not* sufficient to establish proper-name status.
GENERIC_KATAKANA = {
    "パソコン", "ストレス", "データ", "モデル", "システム", "コンピュータ",
    "シミュレーション", "プログラム", "エネルギー", "コスト", "フィルタ",
    "オブジェクト", "カテゴリ", "ウィルス", "ウイルス", "ロボット",
    "アニメ", "マイクロマシン", "ベクタ", "ピンホールカメラ",
    "スナップショット", "コーパス", "サーバ", "ネットワーク",
}
SENTENCE_END = re.compile(r"(?<=[。！？!?])")
QUOTED = re.compile(r"[『「]([^『』「」\n]{2,32})[』」]")
KATAKANA = re.compile(r"(?<![ァ-ヶー])[ァ-ヶー]{4,}(?![ァ-ヶー])")
LATIN = re.compile(r"(?<![A-Za-z])[A-Z][A-Za-z0-9_-]{2,}(?![A-Za-z])")
SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class DirectHit:
    entity: str
    sentence: str
    source: str
    line_number: int


def sentences_from_line(line: str):
    """One genuine source sentence, preserving original words and punctuation."""
    line = line.strip()
    if not line or line.startswith("@") or line.startswith("#"):
        return []
    parts = SENTENCE_END.split(line)
    return [p.strip() for p in parts if p.strip()]


def normalized(text: str) -> str:
    return SPACE.sub("", text).casefold()


def candidate_names(text: str, known_names=()):
    """Build a lexicon; auto candidates require distinctive proper-name form."""
    pool = set(SEED_NAMES) | set(known_names)
    for match in QUOTED.finditer(text):
        pool.add(match.group(1).strip())
    for match in KATAKANA.finditer(text):
        name = match.group(0)
        if name not in GENERIC_KATAKANA and len(name) >= 5:
            pool.add(name)
    for match in LATIN.finditer(text):
        pool.add(match.group(0))
    return {p for p in pool if p.strip() and len(p.strip()) >= 2}


class NagatoDirect:
    """Read original lines, index literal entity occurrences, return one sentence."""

    def __init__(self, source: Path, *, entity_file: Path | None = None):
        self.source = Path(source)
        self.entity_file = Path(entity_file) if entity_file is not None else None
        self.entries: list[tuple[int, str]] = []
        self.names: set[str] = set()
        self.available = False
        self.refresh()

    def refresh(self):
        self.available = self.source.is_file()
        self.entries = []
        self.names = set()
        if not self.available:
            return
        text = self.source.read_text(encoding="utf-8-sig")
        explicit = []
        if self.entity_file is not None and self.entity_file.is_file():
            explicit = [s.strip() for s in self.entity_file.read_text(
                encoding="utf-8-sig").splitlines()
                if s.strip() and not s.lstrip().startswith("#")]
        self.names = candidate_names(text, explicit)
        for line_number, line in enumerate(text.splitlines(), 1):
            for sentence in sentences_from_line(line):
                self.entries.append((line_number, sentence))

    def find(self, question: str) -> DirectHit | None:
        if not self.available:
            return None
        q = normalized(question)
        # Do not treat very broad greetings or short content as an entity.
        if len(q) < 2:
            return None
        names = [
            name for name in self.names
            if normalized(name) in q
        ]
        if not names:
            return None
        # Prefer the longest matching name, followed by original file order.
        names.sort(key=lambda name: (-len(normalized(name)), name))
        for name in names:
            needle = normalized(name)
            for line_number, sentence in self.entries:
                if needle in normalized(sentence):
                    return DirectHit(
                        entity=name,
                        sentence=sentence,
                        source=self.source.name,
                        line_number=line_number,
                    )
        return None
