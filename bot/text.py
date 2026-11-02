"""Text normalisation, a tiny stemmer and a tokenizer.

LEARNING NOTES
--------------
* A bag-of-words model only sees *tokens*, so what we do to the text before
  tokenizing decides what the model can generalise over.
* Numbers are replaced by a placeholder (``numtok``) so "17 + 25" looks like
  "5 + 3" to the model.  Without this, every unseen number is an unknown word.
* Math symbols are mapped to words for the same reason ("+" -> "plus").
* The stemmer is deliberately naive.  Swap it for NLTK's PorterStemmer or a
  lemmatizer and see whether accuracy changes (exercise in the README).
"""
import re

NUM_TOKEN = "numtok"

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

_SYMBOLS = {
    "+": " plus ",
    "*": " times ",
    "\u00d7": " times ",
    "/": " divided by ",
    "\u00f7": " divided by ",
    "^": " power ",
    "%": " percent ",
}

def normalize(text: str) -> str:
    """Lower-case and map numbers / math symbols to stable tokens."""
    s = text.lower()
    s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " minus ", s)
    s = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " times ", s)
    for symbol, word in _SYMBOLS.items():
        s = s.replace(symbol, word)
    s = re.sub(r"\d+(?:\.\d+)?", f" {NUM_TOKEN} ", s)
    return s


def stem(word: str) -> str:
    """Very small suffix-stripping stemmer (good enough for a toy bot)."""
    if word.endswith("'s"):
        word = word[:-2]
    if len(word) <= 3 or "'" in word or word == NUM_TOKEN:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


# Words that carry little meaning.  Only used for retrieval (knowledge.py),
# NOT for intent classification, where words like "what" and "how" matter.
_STOP_RAW = """a an the is are was were be been am do does did of to in on at for with and
    or but if it its this that these those i me my you your we our they them he she his
    her what which who whom how when where why can could would should will shall
    please tell about as by from so than then there here""".split()
STOPWORDS = frozenset(_STOP_RAW) | frozenset(stem(w) for w in _STOP_RAW)


def tokenize(text: str) -> list:
    """Text -> list of stemmed tokens."""
    return [stem(w) for w in _WORD.findall(normalize(text))]


def content_tokens(text: str) -> list:
    """Tokens without stopwords / number placeholders (used for retrieval)."""
    return [t for t in tokenize(text) if t not in STOPWORDS and t != NUM_TOKEN]
