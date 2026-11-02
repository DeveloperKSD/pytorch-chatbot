"""Encoders turn a sentence into a fixed-size vector the neural net can eat.

Two interchangeable implementations share the same interface:

* ``BagOfWordsEncoder``  - counts which known words / word pairs appear.
  Fast, no downloads, but only understands words it saw in training.
* ``SentenceTransformerEncoder`` - pretrained sentence embeddings.  Understands
  paraphrases ("I need a brolly" ~ "will it rain?") because similar meanings
  land close together in vector space.  Needs ``pip install sentence-transformers``
  and a one-time model download.

Interface: ``fit(texts)``, ``encode(texts) -> float32 array``, ``dim``, ``state()``.
"""
import numpy as np

from .text import tokenize


class BagOfWordsEncoder:
    name = "bow"

    def __init__(self, ngram_max=2, vocab=None):
        self.ngram_max = ngram_max
        self.vocab = dict(vocab or {})

    def _features(self, text):
        toks = tokenize(text)
        feats = list(toks)
        if self.ngram_max >= 2:
            feats += [a + "_" + b for a, b in zip(toks, toks[1:])]
        return feats

    def fit(self, texts):
        vocab = {}
        for t in texts:
            for f in self._features(t):
                vocab.setdefault(f, len(vocab))
        self.vocab = vocab
        return self

    @property
    def dim(self):
        return len(self.vocab)

    def encode(self, texts):
        X = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f in self._features(t):
                j = self.vocab.get(f)
                if j is not None:
                    X[i, j] = 1.0
        return X

    def state(self):
        return {"kind": "bow", "ngram_max": self.ngram_max, "vocab": self.vocab}


class SentenceTransformerEncoder:
    name = "sbert"

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dim(self):
        return int(self.model.get_sentence_embedding_dimension())

    def fit(self, texts):  # pretrained: nothing to fit
        return self

    def encode(self, texts):
        vecs = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)

    def state(self):
        return {"kind": "sbert", "model_name": self.model_name}


def make_encoder(kind="bow"):
    if kind == "bow":
        return BagOfWordsEncoder()
    if kind == "sbert":
        return SentenceTransformerEncoder()
    raise ValueError(f"Unknown encoder '{kind}' (use 'bow' or 'sbert')")


def encoder_from_state(state):
    if state["kind"] == "bow":
        return BagOfWordsEncoder(state.get("ngram_max", 2), state["vocab"])
    if state["kind"] == "sbert":
        return SentenceTransformerEncoder(state["model_name"])
    raise ValueError(f"Unknown encoder state {state.get('kind')}")
