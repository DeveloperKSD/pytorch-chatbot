"""Optional LLM fallback - the "hybrid" route.

The fast intent classifier handles everything it is confident about.  Anything
else (unknown questions, open chat) can be passed to a language model, together
with recent conversation history and any passages retrieved from knowledge/.

Enable with an environment variable (default is OFF, the bot works fine without):

    BOT_LLM=ollama      # local model, free.  Install Ollama, then `ollama pull llama3.2`
                        # optional: OLLAMA_MODEL=llama3.2  OLLAMA_HOST=http://localhost:11434
    BOT_LLM=anthropic   # needs ANTHROPIC_API_KEY.  optional: ANTHROPIC_MODEL=claude-sonnet-5-5

Only the standard library is used (urllib), so there is nothing extra to install.
All failures return None and the bot falls back to a canned "I don't know" reply.
NOTE: these network calls are not exercised by the unit tests.
"""
import json
import os
import urllib.request

SYSTEM_PROMPT = (
    "You are Sam, a friendly assistant inside a small chatbot built for learning. "
    "Answer concisely (1-4 sentences). If you are unsure, say so instead of guessing."
)


def _build_messages(history, max_turns=8):
    """bot/user history -> alternating chat messages that start with 'user'."""
    msgs = []
    for role, text in history[-max_turns:]:
        role = "user" if role == "user" else "assistant"
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n" + text
        else:
            msgs.append({"role": role, "content": text})
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


def _post_json(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class LLMFallback:
    def __init__(self, provider, model=None, timeout=60):
        self.provider = provider
        self.timeout = timeout
        self.last_error = None
        if provider == "ollama":
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
            self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        elif provider == "anthropic":
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        else:
            raise ValueError(f"Unknown LLM provider '{provider}'")

    @classmethod
    def from_env(cls):
        provider = os.getenv("BOT_LLM", "none").strip().lower()
        if provider in ("", "none", "off", "0"):
            return None
        return cls(provider)

    def reply(self, history, context=None):
        """Return a string or None.  ``context`` = list of retrieved passages."""
        system = SYSTEM_PROMPT
        if context:
            notes = "\n".join(f"- {c}" for c in context)
            system += "\n\nUse these notes if they are relevant to the question:\n" + notes
        messages = _build_messages(history)
        if not messages:
            return None
        try:
            if self.provider == "ollama":
                data = _post_json(
                    f"{self.host}/api/chat",
                    {"model": self.model, "stream": False,
                     "messages": [{"role": "system", "content": system}] + messages},
                    {"Content-Type": "application/json"}, self.timeout)
                return (data.get("message", {}).get("content") or "").strip() or None
            if not self.api_key:
                self.last_error = "ANTHROPIC_API_KEY is not set"
                return None
            data = _post_json(
                "https://api.anthropic.com/v1/messages",
                {"model": self.model, "max_tokens": 500, "system": system, "messages": messages},
                {"Content-Type": "application/json", "x-api-key": self.api_key,
                 "anthropic-version": "2023-06-01"}, self.timeout)
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            return text.strip() or None
        except Exception as exc:  # network down, bad key, model missing ...
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
