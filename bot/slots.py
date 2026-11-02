"""Slot (entity) extraction with plain regular expressions.

An *intent* says what the user wants ("weather"); *slots* are the details needed
to do it (city, date).  Each extractor returns a value or None.

``expecting=True`` means "we just asked the user for exactly this slot", so we
can be more forgiving: a bare "tbilisi" is accepted as a city even though it is
not in our list.

LEARNING NOTES
--------------
* Regex extraction is brittle but transparent.  The natural upgrade is a trained
  NER model (spaCy) or asking an LLM for JSON - compare them on your own data.
* Dates are normalised to ISO strings so the rest of the code never parses
  "tomorrow" again.
"""
import re
from datetime import datetime, timedelta

CITIES = [
    "Mumbai", "Delhi", "New Delhi", "Bangalore", "Bengaluru", "Chennai", "Kolkata", "Pune",
    "Hyderabad", "London", "Paris", "Berlin", "Madrid", "Rome", "Tokyo", "Seoul", "Beijing",
    "Shanghai", "Singapore", "Dubai", "Sydney", "Melbourne", "New York", "Los Angeles",
    "San Francisco", "Chicago", "Toronto", "Vancouver", "Mexico City", "Cairo", "Lagos",
    "Nairobi", "Moscow", "Istanbul", "Bangkok", "Jakarta", "Amsterdam", "Lisbon", "Dublin",
]

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DATE_WORDS = {"today", "tomorrow", "tonight", "yesterday", "now", "morning", "evening"} | set(WEEKDAYS)
_CANCEL = re.compile(r"\b(cancel|never\s*mind|nevermind|forget it|stop|abort)\b", re.I)


def is_cancel(text):
    return bool(_CANCEL.search(text))


def describe_date(iso, now=None):
    """'2026-11-02' -> 'tomorrow' / 'today' / 'on Monday'."""
    now = now or datetime.now()
    d = datetime.strptime(iso, "%Y-%m-%d").date()
    delta = (d - now.date()).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta == -1:
        return "yesterday"
    return "on " + d.strftime("%A %d %B")


# ------------------------------------------------------------------ extractors
def _city(text, expecting, now):
    low = text.lower()
    for c in sorted(CITIES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(c.lower()) + r"\b", low):
            return c
    m = re.search(r"\b(?:in|at|for|near)\s+([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2})", text)
    if m and m.group(1).lower() not in _DATE_WORDS:
        return m.group(1)
    if expecting and not is_cancel(text):
        words = re.findall(r"[A-Za-z][\w'-]*", text)
        if 1 <= len(words) <= 3 and not any(w.lower() in _DATE_WORDS for w in words):
            return " ".join(w.capitalize() for w in words)
    return None


def _date(text, expecting, now):
    now = now or datetime.now()
    low = text.lower()
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", low)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date().isoformat()
        except ValueError:
            return None
    if "day after tomorrow" in low:
        return (now + timedelta(days=2)).date().isoformat()
    if "tomorrow" in low:
        return (now + timedelta(days=1)).date().isoformat()
    if "today" in low or "tonight" in low:
        return now.date().isoformat()
    if "yesterday" in low:
        return (now - timedelta(days=1)).date().isoformat()
    m = re.search(r"\b(" + "|".join(WEEKDAYS) + r")\b", low)
    if m:
        target = WEEKDAYS.index(m.group(1))
        ahead = (target - now.weekday()) % 7 or 7  # next occurrence, never "today"
        return (now + timedelta(days=ahead)).date().isoformat()
    return None


def _time(text, expecting, now):
    low = text.lower()
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", low)
    if m:
        hour, minute = int(m.group(1)) % 12, int(m.group(2) or 0)
        if m.group(3) == "pm":
            hour += 12
        return f"{hour:02d}:{minute:02d}"
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", low)
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None


def _number(text, expecting, now):
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else None


def _name(text, expecting, now):
    m = re.search(r"\b(?:my name is|my name's|i am|i'm|im|call me|this is|name is|name's)\s+([A-Za-z][\w'-]*)",
                  text, re.I)
    if m:
        return m.group(1).capitalize()
    if expecting and not is_cancel(text):
        words = re.findall(r"[A-Za-z][\w'-]*", text)
        if 1 <= len(words) <= 2:
            return words[-1].capitalize()
    return None


_NOT_ITEMS = {"something", "anything", "stuff", "it", "a", "an"}


def _item(text, expecting, now):
    """Free-text to-do item: 'add buy milk to my todo list' -> 'buy milk'."""
    t = text.strip()
    if expecting:
        return t.strip(" .!?") or None
    t = re.sub(r"^(?:please\s+)?(?:add|put|remind me to|remember to|note down|note|new task|new todo)\b\s*[:\-]?\s*",
               "", t, flags=re.I)
    t = re.sub(r"^(?:a\s+|an\s+|the\s+)?(?:new\s+)?(?:task|todo|to-do|item)\b\s*[:\-]?\s*", "", t, flags=re.I)
    t = re.sub(r"\s+(?:to|on|onto|in|into)\s+(?:my|the)\s+(?:(?:to-?do|task|tasks|shopping)\s*)?(?:list)?\s*[.!?]*$",
               "", t, flags=re.I)
    t = t.strip(" .!?")
    return t if t and t.lower() not in _NOT_ITEMS else None


_EXTRACTORS = {
    "city": _city, "date": _date, "time": _time,
    "number": _number, "name": _name, "item": _item,
}


def extract(slot_type, text, expecting=False, now=None):
    fn = _EXTRACTORS.get(slot_type)
    if fn is None:
        raise ValueError(f"No extractor for slot type '{slot_type}'")
    return fn(text, expecting, now or datetime.now())


def resolve_default(slot_type, default, now=None):
    """Turn a default from intents.json ('today', 'tomorrow', literal) into a value."""
    now = now or datetime.now()
    if slot_type == "date":
        offsets = {"today": 0, "tomorrow": 1}
        if default in offsets:
            return (now + timedelta(days=offsets[default])).date().isoformat()
    return default


def extract_all(text, now=None):
    """Debug helper: run every extractor on a sentence."""
    out = {}
    for name in _EXTRACTORS:
        value = extract(name, text, now=now)
        if value is not None:
            out[name] = value
    return out
