"""The brain that decides WHAT TO DO with each message.

Per message, in order:

 1. Answering a "did you mean ...?" question?          -> yes / no handling
 2. In the middle of slot filling ("Which city?")?     -> take the answer, ask next / finish
 3. Otherwise classify the sentence, then route on confidence:

        confidence >= hi          -> run the intent (answer / action / ask for slots)
        knowledge match >= thr    -> answer from your notes (RAG if an LLM is enabled)
        confidence >= lo          -> "Did you mean X?"  (clarification)
        otherwise                 -> LLM if enabled, else an honest "I don't know" (logged)

Intent options understood in intents.json:
    patterns, responses, description
    slots           {name: {type, prompt, default?, remember?}}
    action          name of a function registered in actions.py
    context_set     topic to remember after this intent ("joke")
    context_filter  intent only eligible while that topic is active
    context_hint    what to say if a context_filter intent is used without its context
    route           "fallback" (out-of-scope examples) or "knowledge" (ask the docs)
"""
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from . import slots as slotlib
from .actions import ACTIONS, ActionContext

YES = {"yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "correct", "right", "exactly", "ya"}
NO = {"no", "n", "nope", "nah", "wrong", "incorrect"}


@dataclass
class Reply:
    text: str
    intent: Optional[str] = None
    confidence: float = 0.0
    route: str = "intent"  # intent | slot | clarify | knowledge | rag | llm | fallback | empty
    debug: dict = field(default_factory=dict)


def first_tokens(text):
    return re.findall(r"[a-z']+", text.lower())


class _SafeDict(dict):
    def __missing__(self, key):
        return ""


class DialogManager:
    def __init__(self, classifier, intents_doc, knowledge=None, llm=None, unknown_log=None,
                 hi=0.75, lo=0.30, kb_threshold=0.35, seed=None, now_fn=None):
        self.classifier = classifier
        self.set_intents(intents_doc)
        self.knowledge = knowledge
        self.llm = llm
        self.unknown_log = unknown_log
        self.hi, self.lo, self.kb_threshold = hi, lo, kb_threshold
        self.rng = random.Random(seed)
        self.now = now_fn or datetime.now

    def set_intents(self, doc):
        self.intents = {i["tag"]: i for i in doc["intents"]}
        self.fallbacks = doc.get("fallback") or [
            "I'm not sure I understood. Could you rephrase?"]

    # ------------------------------------------------------------------ public
    def reply(self, text, session):
        text = (text or "").strip()
        if not text:
            return Reply("Say something and I'll do my best!", route="empty")
        session.add("user", text)
        result = self._route(text, session)
        session.add("bot", result.text)
        return result

    # ----------------------------------------------------------------- routing
    def _route(self, text, session):
        if session.suggestion:
            tag, original = session.suggestion
            session.suggestion = None
            words = first_tokens(text)
            if words and words[0] in YES and len(words) <= 3:
                return self._handle_intent(tag, original, session, 1.0, "intent", {"via": "clarify-yes"})
            if words and words[0] in NO and len(words) <= 3:
                return Reply("No problem - could you rephrase that? "
                             "(Or teach me with /teach <intent> in the terminal.)", route="clarify")
            # anything else: the user ignored the question, treat as a fresh message

        if session.pending:
            result = self._continue_pending(text, session)
            if result:
                return result

        return self._classify_and_answer(text, session)

    def _eligible(self, tag, session):
        wanted = self.intents.get(tag, {}).get("context_filter")
        return wanted is None or wanted == session.context

    def _ranked(self, text, session):
        probs = self.classifier.predict(text)
        return [(t, p) for t, p in probs if self._eligible(t, session)]

    def _classify_and_answer(self, text, session):
        raw_tag, raw_p = self.classifier.predict(text, 1)[0]
        if raw_p >= self.hi and not self._eligible(raw_tag, session):
            hint = self.intents[raw_tag].get("context_hint")
            if hint:  # e.g. "another one" without a joke before it
                return Reply(hint, intent=raw_tag, confidence=raw_p, route="clarify",
                             debug={"blocked_by_context": raw_tag})
        ranked = self._ranked(text, session)
        top_tag, top_p = ranked[0]
        debug = {"top": ranked[:3], "context": session.context}
        intent = self.intents[top_tag]
        route_kind = intent.get("route")

        if top_p >= self.hi and route_kind is None:
            return self._handle_intent(top_tag, text, session, top_p, "intent", debug)

        if top_p >= self.hi and route_kind == "knowledge":
            return self._unknown_path(text, session, ranked, debug, force_kb=True)

        if top_p >= self.hi and route_kind == "fallback":
            return self._unknown_path(text, session, ranked, debug)

        # not confident enough: try the knowledge base before bothering the user
        if self.knowledge:
            hits = self.knowledge.search(text)
            if hits and hits[0].score >= self.kb_threshold:
                return self._answer_from_knowledge(text, session, hits, debug)

        if top_p >= self.lo and route_kind is None:
            session.suggestion = (top_tag, text)
            session.last_unknown = text
            label = intent.get("description") or top_tag.replace("_", " ")
            return Reply(f"I think you're asking about \"{label}\" - is that right? (yes / no)",
                         intent=top_tag, confidence=top_p, route="clarify", debug=debug)

        return self._unknown_path(text, session, ranked, debug, kb_done=True)

    # ------------------------------------------------------------ unknown path
    def _unknown_path(self, text, session, ranked, debug, force_kb=False, kb_done=False):
        if self.knowledge and not kb_done:
            hits = self.knowledge.search(text)
            threshold = 0.10 if force_kb else self.kb_threshold
            if hits and hits[0].score >= threshold:
                return self._answer_from_knowledge(text, session, hits, debug)
        session.last_unknown = text
        if self.unknown_log:
            self.unknown_log.add(text, ranked[:3])
        if force_kb and not self.llm:
            return Reply("That sounds like a question about how I work, but I couldn't find it in my notes "
                         "(knowledge/). You can add a .md file there.", route="fallback", debug=debug)
        if self.llm:
            answer = self.llm.reply(session.history)
            if answer:
                return Reply(answer, route="llm", debug=debug)
        return Reply(self.rng.choice(self.fallbacks), route="fallback", debug=debug)

    def _answer_from_knowledge(self, text, session, hits, debug):
        debug = dict(debug, kb=[(h.source, round(h.score, 2)) for h in hits])
        if self.llm:
            answer = self.llm.reply(session.history, context=[h.text for h in hits])
            if answer:
                return Reply(answer, route="rag", debug=debug)
        best = hits[0]
        snippet = best.text if len(best.text) <= 450 else best.text[:447].rstrip() + "..."
        return Reply(f"From my notes ({best.source}): {snippet}", route="knowledge",
                     confidence=best.score, debug=debug)

    # -------------------------------------------------------- intents & slots
    def _handle_intent(self, tag, text, session, conf, route, debug):
        intent = self.intents[tag]
        now = self.now()
        values, missing = {}, []
        for name, spec in intent.get("slots", {}).items():
            value = slotlib.extract(spec["type"], text, now=now)
            if value is None and spec.get("remember"):
                value = session.slot_memory.get(name)          # reuse from earlier turns
            if value is None and "default" in spec:
                value = slotlib.resolve_default(spec["type"], spec["default"], now)
            if value is None:
                missing.append(name)
            else:
                values[name] = value
        if missing:
            session.pending = {"tag": tag, "slots": values, "missing": missing, "asked": 0, "text": text}
            prompt = intent["slots"][missing[0]].get("prompt", f"What is the {missing[0]}?")
            return Reply(prompt, intent=tag, confidence=conf, route="slot", debug=debug)
        return self._finish(intent, values, text, session, conf, route, debug)

    def _continue_pending(self, text, session):
        """Return a Reply if the message was consumed, or None to treat it as a new topic."""
        p = session.pending
        intent = self.intents[p["tag"]]
        if slotlib.is_cancel(text):
            session.pending = None
            return Reply("No problem, cancelled.", intent=p["tag"], route="slot")

        name = p["missing"][0]
        spec = intent["slots"][name]

        # free-text slots swallow anything; for others, a confident different intent = topic switch
        if spec["type"] != "item":
            ranked = self._ranked(text, session)
            if ranked[0][1] >= self.hi and ranked[0][0] != p["tag"] and not self.intents[ranked[0][0]].get("route"):
                session.pending = None
                return None

        value = slotlib.extract(spec["type"], text, expecting=True, now=self.now())
        if value is None:
            p["asked"] += 1
            if p["asked"] >= 2:
                session.pending = None
                return Reply("I still couldn't get that, so let's drop it. Ask me again any time.",
                             intent=p["tag"], route="slot")
            return Reply("Sorry, I didn't catch that. " + spec.get("prompt", f"What is the {name}?"),
                         intent=p["tag"], route="slot")

        p["slots"][name] = value
        p["missing"].pop(0)
        if p["missing"]:
            nxt = p["missing"][0]
            return Reply(intent["slots"][nxt].get("prompt", f"What is the {nxt}?"),
                         intent=p["tag"], route="slot")
        session.pending = None
        return self._finish(intent, p["slots"], p.get("text", text), session, 1.0, "slot", {})

    def _finish(self, intent, values, text, session, conf, route, debug):
        tag = intent["tag"]
        for name, spec in intent.get("slots", {}).items():
            if spec.get("remember") and name in values:
                session.slot_memory[name] = values[name]
        session.pending = None
        if intent.get("action"):
            body = ACTIONS[intent["action"]](ActionContext(text, values, session, self.now()))
        else:
            template = self.rng.choice(intent["responses"])
            mapping = _SafeDict({"name": session.facts.get("name", "friend"), **values})
            body = template.format_map(mapping)
        session.context = intent.get("context_set")
        session.last_intent = tag
        return Reply(body, intent=tag, confidence=conf, route=route, debug=dict(debug, slots=values))
