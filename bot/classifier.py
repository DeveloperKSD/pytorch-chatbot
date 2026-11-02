"""IntentClassifier: encoder + neural net + label names, with save/load.

LEARNING NOTES
--------------
* Data augmentation (random word dropout) multiplies a tiny dataset and makes
  the model less dependent on any single word.
* A validation split + early stopping stops training when the model starts to
  memorise instead of generalise.
* ``predict`` returns a probability for *every* intent.  The dialog manager uses
  those numbers to decide: answer, ask for clarification, or admit ignorance.
* If a sentence contains no known words at all, the encoded vector is all zeros
  and the network would output whatever its biases favour.  We return a uniform
  distribution instead ("I have no evidence") - a simple out-of-vocabulary guard.
"""
import json
import os
import random

import numpy as np

from .encoders import encoder_from_state, make_encoder
from .models import make_backend, weights_filename


def samples_from_intents(doc):
    return [(p, i["tag"]) for i in doc["intents"] for p in i.get("patterns", [])]


def tags_from_intents(doc):
    return [i["tag"] for i in doc["intents"] if i.get("patterns")]


def split_samples(samples, val_split, seed=0):
    """Per-intent split. Intents with fewer than 5 examples stay entirely in train."""
    if val_split <= 0:
        return list(samples), []
    rng = random.Random(seed)
    by_tag = {}
    for s in samples:
        by_tag.setdefault(s[1], []).append(s)
    train, val = [], []
    for items in by_tag.values():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_split)) if len(items) >= 5 else 0
        val += items[:n_val]
        train += items[n_val:]
    return train, val


def augment_text(text, rng, p=0.2):
    """Randomly drop words (keeps at least one)."""
    words = text.split()
    if len(words) <= 2:
        return text
    kept = [w for w in words if rng.random() > p]
    return " ".join(kept) if kept else text


class IntentClassifier:
    def __init__(self, encoder, backend, tags, config=None, history=None):
        self.encoder = encoder
        self.backend = backend
        self.tags = list(tags)
        self.config = config or {}
        self.history = history or {}

    # ------------------------------------------------------------------ train
    @classmethod
    def fit(cls, train, val, tags, encoder="bow", backend="auto", hidden=64, dropout=0.2,
            epochs=150, lr=3e-3, batch_size=16, patience=40, augment=3, seed=0, log=print):
        tag_idx = {t: i for i, t in enumerate(tags)}
        rng = random.Random(seed)

        enc = make_encoder(encoder)
        enc.fit([t for t, _ in train])

        texts, labels = [], []
        for text, tag in train:
            texts.append(text)
            labels.append(tag_idx[tag])
            for _ in range(augment):
                texts.append(augment_text(text, rng))
                labels.append(tag_idx[tag])
        X = enc.encode(texts)
        y = np.array(labels, dtype=np.int64)

        Xv = yv = None
        if val:
            Xv = enc.encode([t for t, _ in val])
            yv = np.array([tag_idx[tag] for _, tag in val], dtype=np.int64)

        be = make_backend(backend, enc.dim, hidden, len(tags), dropout, seed)
        if log:
            log(f"[train] encoder={enc.name} backend={be.name} samples={len(texts)} "
                f"(orig {len(train)}, val {len(val)}) features={enc.dim} intents={len(tags)}")
        history = be.fit(X, y, Xv, yv, epochs=epochs, lr=lr, batch_size=batch_size,
                         patience=patience, log=log)
        config = {"encoder": enc.name, "backend": be.name, "hidden": hidden, "dropout": dropout,
                  "epochs": epochs, "lr": lr, "augment": augment, "seed": seed}
        return cls(enc, be, tags, config, history)

    # ---------------------------------------------------------------- predict
    def predict(self, text, top_k=None):
        """Return [(tag, probability), ...] sorted from most to least likely."""
        X = self.encoder.encode([text])
        if X.any():
            probs = self.backend.predict_proba(X)[0]
        else:  # nothing recognised -> no evidence -> uniform
            probs = np.full(len(self.tags), 1.0 / len(self.tags))
        order = np.argsort(-probs)
        if top_k:
            order = order[:top_k]
        return [(self.tags[i], float(probs[i])) for i in order]

    def accuracy(self, samples):
        if not samples:
            return float("nan")
        hits = sum(self.predict(t, 1)[0][0] == tag for t, tag in samples)
        return hits / len(samples)

    # ------------------------------------------------------------- save / load
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        weights = weights_filename(self.backend.name)
        self.backend.save(os.path.join(directory, weights))
        meta = {
            "tags": self.tags,
            "config": self.config,
            "encoder": self.encoder.state(),
            "backend": self.backend.name,
            "weights": weights,
            "dims": {"in": self.encoder.dim, "hidden": self.config.get("hidden", 64),
                     "out": len(self.tags)},
            "dropout": self.config.get("dropout", 0.2),
            "history": {k: [round(float(x), 5) for x in v] for k, v in self.history.items()},
        }
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        enc = encoder_from_state(meta["encoder"])
        d = meta["dims"]
        be = make_backend(meta["backend"], d["in"], d["hidden"], d["out"], meta.get("dropout", 0.2))
        be.load(os.path.join(directory, meta["weights"]))
        return cls(enc, be, meta["tags"], meta.get("config"), meta.get("history"))
