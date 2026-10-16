"""Knowledge-base retrieval (FAQs, user manuals, troubleshooting guides).

Each markdown file is split into chunks at `## ` headings so a search returns the
specific section that answers the question rather than a whole document. Ranking is
TF-IDF with unigrams + bigrams and cosine similarity: no API calls, no quota, and
good enough for a corpus of this size. Swapping in embeddings later only means
replacing `KnowledgeBase.search`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .config import KB_DIR


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    section: str
    text: str
    doc_type: str  # faq | manual | troubleshooting


def _doc_type(name: str) -> str:
    if name.startswith("faq_"):
        return "faq"
    if name.startswith("manual_"):
        return "manual"
    return "troubleshooting"


def load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", raw, flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem
        for part in re.split(r"^## ", raw, flags=re.MULTILINE)[1:]:
            heading, _, body = part.partition("\n")
            chunks.append(Chunk(path.stem, title, heading.strip(), body.strip(), _doc_type(path.stem)))
    return chunks


class KnowledgeBase:
    def __init__(self, kb_dir: Path = KB_DIR):
        self.chunks = load_chunks(kb_dir)
        # Title + heading are repeated so they weigh more than body text.
        corpus = [f"{c.doc_title} {c.doc_title} {c.section} {c.section} {c.text}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> list[dict]:
        if not query.strip():
            return []
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [
            {"doc_id": self.chunks[i].doc_id, "title": self.chunks[i].doc_title,
             "section": self.chunks[i].section, "type": self.chunks[i].doc_type,
             "score": round(float(scores[i]), 3), "content": self.chunks[i].text}
            for i in ranked if scores[i] >= min_score
        ]


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()
