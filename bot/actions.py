"""Actions = Python functions an intent can trigger (the bot's "tools").

Register a new one with ``@action("name")`` and reference it from an intent in
intents.json with ``"action": "name"``.  The function gets an ``ActionContext``
and returns the reply text.

LEARNING NOTES
--------------
* This is the same idea as "tool use" / "function calling" in LLM agents - here
  the *router* (our classifier) picks the tool instead of a language model.
* ``safe_eval`` shows how to evaluate user-supplied math WITHOUT ``eval()``,
  which would let anyone run arbitrary code.  We parse into an AST and only
  allow a whitelist of node types.
"""
import ast
import hashlib
import operator
import random
import re
from dataclasses import dataclass
from datetime import datetime

from .slots import describe_date

ACTIONS = {}


@dataclass
class ActionContext:
    text: str
    slots: dict
    session: object
    now: datetime


def action(name):
    def register(fn):
        ACTIONS[name] = fn
        return fn
    return register


# ------------------------------------------------------------------ calculator
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expr):
    """Evaluate a math expression using only numbers and + - * / % ** //."""
    if len(expr) > 100:
        raise ValueError("expression too long")
    tree = ast.parse(expr.strip(), mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    return ev(tree)


_WORD_OPS = [
    (r"\bto the power of\b|\bpower\b", " ** "),
    (r"\bmultiplied by\b|\btimes\b", " * "),
    (r"\bdivided by\b|\bover\b", " / "),
    (r"\bplus\b", " + "),
    (r"\bminus\b", " - "),
    (r"\bmod(?:ulo)?\b", " % "),
    (r"\^", " ** "),
]


def extract_expression(text):
    """'what is 5 plus 3?' -> '5  +  3'"""
    s = text.lower()
    for pattern, repl in _WORD_OPS:
        s = re.sub(pattern, repl, s)
    s = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " * ", s)
    s = s.replace("\u00d7", " * ").replace("\u00f7", " / ")
    candidates = re.findall(r"[\d.()\s+\-*/%]+", s)
    candidates = [c.strip() for c in candidates if re.search(r"\d", c) and re.search(r"[+\-*/%]", c)]
    return max(candidates, key=len) if candidates else None


def _fmt_number(x):
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


@action("calc")
def _calc(ctx):
    expr = extract_expression(ctx.text)
    if not expr:
        return "I couldn't find a calculation in that. Try something like '12 * (3 + 4)'."
    try:
        value = safe_eval(expr)
    except ZeroDivisionError:
        return "I can't divide by zero - that's undefined."
    except (ValueError, SyntaxError, OverflowError, TypeError):
        return "I couldn't evaluate that expression safely."
    return f"{' '.join(expr.split())} = {_fmt_number(value)}"


# ------------------------------------------------------------------ small tools
@action("datetime")
def _datetime(ctx):
    return ctx.now.strftime("It's %H:%M on %A, %d %B %Y.")


_JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "I told my neural network a joke. It didn't get it, but it did overfit to the punchline.",
    "There are 10 kinds of people: those who understand binary and those who don't.",
    "Why was the math book sad? It had too many problems.",
    "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
    "Why did the gradient go to therapy? It kept vanishing when things got deep.",
    "How many programmers does it take to change a light bulb? None - that's a hardware problem.",
    "I would tell you a UDP joke, but you might not get it.",
]


@action("joke")
def _joke(ctx):
    return random.choice(_JOKES)


@action("weather")
def _weather(ctx):
    """DEMO DATA: deterministic fake weather so the slot-filling flow is testable.

    EXERCISE: replace with a real API call (e.g. Open-Meteo needs no key).
    """
    city = ctx.slots["city"]
    date_iso = ctx.slots.get("date") or ctx.now.date().isoformat()
    seed = int(hashlib.md5(f"{city.lower()}|{date_iso}".encode()).hexdigest(), 16)
    rnd = random.Random(seed)
    cond = rnd.choice(["sunny", "partly cloudy", "cloudy", "light rain", "windy", "clear skies"])
    temp = rnd.randint(8, 34)
    when = describe_date(date_iso, ctx.now)
    return f"Weather in {city} {when}: {cond}, around {temp}\u00b0C. (demo data - plug in a real API in actions.py)"


# ------------------------------------------------------------------ memory tools
@action("remember_name")
def _remember_name(ctx):
    name = ctx.slots["name"]
    ctx.session.facts["name"] = name
    return f"Nice to meet you, {name}! I'll remember that."


@action("recall_name")
def _recall_name(ctx):
    name = ctx.session.facts.get("name")
    if name:
        return f"You told me your name is {name}."
    return "I don't know your name yet - tell me with 'my name is ...'."


@action("todo_add")
def _todo_add(ctx):
    item = ctx.slots["item"]
    ctx.session.todos.append(item)
    n = len(ctx.session.todos)
    return f"Added \"{item}\". You now have {n} item{'s' if n != 1 else ''} on your list."


@action("todo_list")
def _todo_list(ctx):
    todos = ctx.session.todos
    if not todos:
        return "Your list is empty. Try 'add buy milk to my list'."
    lines = "\n".join(f"  {i}. {t}" for i, t in enumerate(todos, 1))
    return f"Here's your list:\n{lines}"
