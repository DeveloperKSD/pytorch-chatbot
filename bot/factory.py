"""Glue code shared by the CLI, web UI, evaluation script and tests."""
import json
import os
import shutil

from .classifier import (IntentClassifier, samples_from_intents, split_samples,
                         tags_from_intents)
from .dialog import DialogManager
from .knowledge import KnowledgeBase
from .llm import LLMFallback
from .memory import UnknownLog


def load_intents(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_intents(path, doc):
    if os.path.exists(path):
        shutil.copyfile(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def train_and_save(intents_path, model_dir, encoder="bow", backend="auto", hidden=64,
                   epochs=150, lr=3e-3, val_split=0.2, augment=3, seed=0, log=print):
    """Two-phase training (a standard ML pattern):

    1. Hold out part of the data, train with early stopping and REPORT accuracy.
    2. Refit on ALL the data (a bot should learn from every example you wrote)
       for a fixed number of epochs and save that model.
    """
    doc = load_intents(intents_path)
    samples, tags = samples_from_intents(doc), tags_from_intents(doc)
    kw = dict(encoder=encoder, backend=backend, hidden=hidden, epochs=epochs, lr=lr,
              augment=augment, seed=seed)

    held_out = None
    train, val = split_samples(samples, val_split, seed)
    if val:
        if log:
            log(f"[train] phase 1/2: measuring accuracy ({len(train)} train / {len(val)} held-out sentences)")
        probe = IntentClassifier.fit(train, val, tags, log=log, **kw)
        held_out = probe.accuracy(val)
        if log:
            log(f"[train] held-out accuracy: {held_out:.1%}   (optimistic: the same split picked the best epoch;"
                f" run evaluate.py for a fair estimate)")
    if log:
        log(f"[train] phase {'2/2' if val else '1/1'}: fitting final model on all {len(samples)} sentences")
    clf = IntentClassifier.fit(samples, [], tags, log=log, **kw)
    if held_out is not None:
        clf.config["held_out_accuracy"] = round(held_out, 4)
    if log:
        log(f"[train] accuracy on its own training data: {clf.accuracy(samples):.1%}")
    clf.save(model_dir)
    if log:
        log(f"[train] saved model to {model_dir}/")
    return clf


def model_exists(model_dir):
    return os.path.exists(os.path.join(model_dir, "meta.json"))


def build_dialog(model_dir="model", intents_path="intents.json", knowledge_dir="knowledge",
                 unknown_log_path="data/unknown.jsonl", use_llm=True, **kwargs):
    doc = load_intents(intents_path)
    clf = IntentClassifier.load(model_dir)
    kb = KnowledgeBase(knowledge_dir) if knowledge_dir and os.path.isdir(knowledge_dir) else None
    if kb is not None and len(kb) == 0:
        kb = None
    llm = LLMFallback.from_env() if use_llm else None
    log = UnknownLog(unknown_log_path) if unknown_log_path else None
    return DialogManager(clf, doc, knowledge=kb, llm=llm, unknown_log=log, **kwargs)


# ------------------------------------------------------------ teaching helpers
def add_pattern(intents_path, tag, text):
    """Append a training sentence to an existing intent. Returns False if tag unknown."""
    doc = load_intents(intents_path)
    for intent in doc["intents"]:
        if intent["tag"] == tag:
            if text not in intent["patterns"]:
                intent["patterns"].append(text)
            save_intents(intents_path, doc)
            return True
    return False


def add_intent(intents_path, tag, text, response):
    """Create a brand-new intent with one pattern and one canned response."""
    doc = load_intents(intents_path)
    doc["intents"].append({"tag": tag, "description": tag.replace("_", " "),
                           "patterns": [text], "responses": [response]})
    save_intents(intents_path, doc)
