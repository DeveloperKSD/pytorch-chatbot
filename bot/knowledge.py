"""A tiny retrieval engine over your own notes (the "R" in RAG).

Drop .md / .txt files into the ``knowledge/`` folder.  Each paragraph becomes a
searchable chunk; a question is matched to chunks with TF-IDF cosine similarity.

* Without an LLM the best chunk is returned as-is (extractive answer).
* With an LLM (see llm.py) the top chunks are handed over as context so the model
  can answer in its own words but grounded in YOUR documents - that is RAG.

LEARNING NOTES
--------------
* TF-IDF weights a word by how often it appears in a chunk (TF) and how rare it
  is across all chunks (IDF), so "slot" matters more than "the".
* Exercise: swap the scoring for embeddings (reuse SentenceTransformerEncoder)
  and compare which retrieves better on paraphrased questions.
"""
import glob
import math
import os
from collections import Counter
from dataclasses import dataclass

from .text import content_tokens


@dataclass
class Hit:
    score: float
    text: str
    source: str


class KnowledgeBase:
    def __init__(self, folder):
        self.chunks = []  # (source, text)
        self._vectors = []
        self._idf = {}
        for path in sorted(glob.glob(os.path.join(folder, "*.md")) + glob.glob(os.path.join(folder, "*.txt"))):
            with open(path, encoding="utf-8") as f:
                self._add_document(os.path.basename(path), f.read())
        self._build_index()

    def __len__(self):
        return len(self.chunks)

    def _add_document(self, source, text):
        heading = ""
        for block in text.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if block.startswith("#"):
                lines = block.splitlines()
                heading = lines[0].lstrip("# ").strip()
                block = "\n".join(lines[1:]).strip()
                if not block:
                    continue
            if len(block) < 20:
                continue
            self.chunks.append((source, f"{heading}: {block}" if heading else block))

    def _build_index(self):
        docs = [Counter(content_tokens(text)) for _, text in self.chunks]
        n = len(docs)
        df = Counter(tok for d in docs for tok in d)
        self._idf = {tok: math.log((n + 1) / (c + 1)) + 1.0 for tok, c in df.items()}
        self._vectors = [self._weigh(d) for d in docs]

    def _weigh(self, counts):
        vec = {t: (1 + math.log(c)) * self._idf.get(t, 0.0) for t, c in counts.items() if t in self._idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query, k=2):
        q = self._weigh(Counter(content_tokens(query)))
        if not q:
            return []
        scored = []
        for (source, text), vec in zip(self.chunks, self._vectors):
            score = sum(w * vec.get(t, 0.0) for t, w in q.items())
            if score > 0:
                scored.append(Hit(score, text, source))
        scored.sort(key=lambda h: -h.score)
        return scored[:k]
