"""Conversation state and the "things I didn't understand" log.

``Session`` is everything the bot remembers about ONE conversation:

* history       - last N turns (also fed to the optional LLM)
* context       - short-term topic ("joke") used to unlock follow-up intents
* pending       - a half-finished task waiting for a missing slot
* suggestion    - a "did you mean ...?" question we are waiting on
* slot_memory   - values reused across turns ("and tomorrow?" keeps the city)
* facts / todos - long-term memory, persisted to disk by the CLI

``UnknownLog`` appends every message the bot could not confidently handle to a
JSONL file.  Reading it tells you exactly which intents to add next - the core of
the "improve from real usage" loop.
"""
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Session:
    history: list = field(default_factory=list)
    context: Optional[str] = None
    pending: Optional[dict] = None
    suggestion: Optional[tuple] = None
    slot_memory: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    todos: list = field(default_factory=list)
    last_intent: Optional[str] = None
    last_unknown: Optional[str] = None

    MAX_HISTORY = 40

    def add(self, role, text):
        self.history.append((role, text))
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]

    def reset_short_term(self):
        self.history, self.context, self.pending = [], None, None
        self.suggestion, self.last_intent, self.last_unknown = None, None, None

    # -- persistence of long-term memory only
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"facts": self.facts, "todos": self.todos, "slot_memory": self.slot_memory}, f, indent=2)

    @classmethod
    def load(cls, path):
        s = cls()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            s.facts = data.get("facts", {})
            s.todos = data.get("todos", [])
            s.slot_memory = data.get("slot_memory", {})
        except (OSError, ValueError):
            pass
        return s


class UnknownLog:
    def __init__(self, path):
        self.path = path

    def add(self, text, top=None):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        row = {"ts": int(time.time()), "text": text,
               "top": [(t, round(p, 3)) for t, p in (top or [])]}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def rows(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except OSError:
            return []

    def most_common(self, n=10):
        return Counter(r["text"].lower() for r in self.rows()).most_common(n)
