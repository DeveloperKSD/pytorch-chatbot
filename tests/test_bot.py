"""Run with:  python -m unittest discover -s tests -v   (from the project root)"""
import os
import sys
import tempfile
import unittest
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from bot import slots
from bot.actions import extract_expression, safe_eval
from bot.classifier import IntentClassifier, samples_from_intents, tags_from_intents
from bot.dialog import DialogManager
from bot.factory import load_intents
from bot.knowledge import KnowledgeBase
from bot.memory import Session, UnknownLog
from bot.models import NumpyMLP, torch_available
from bot.text import tokenize

NOW = datetime(2026, 11, 2, 12, 0)  # a Monday


class TextTests(unittest.TestCase):
    def test_numbers_and_symbols_are_normalised(self):
        self.assertEqual(tokenize("17 + 25"), tokenize("5 plus 3"))

    def test_plural_and_singular_match(self):
        self.assertEqual(tokenize("jokes"), tokenize("joke"))


class CalculatorTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(safe_eval("2 + 3 * 4"), 14)
        self.assertEqual(safe_eval("(3 + 4) * 5"), 35)

    def test_rejects_code(self):
        for bad in ["__import__('os').system('echo hi')", "open('x')", "[1]*10", "'a'*5"]:
            with self.assertRaises((ValueError, SyntaxError, TypeError)):
                safe_eval(bad)

    def test_huge_power_refused(self):
        with self.assertRaises(ValueError):
            safe_eval("9 ** 9 ** 9")

    def test_extract_from_words(self):
        self.assertEqual(safe_eval(extract_expression("what is 5 plus 3?")), 8)
        self.assertEqual(safe_eval(extract_expression("what is 2 to the power of 10")), 1024)


class SlotTests(unittest.TestCase):
    def test_city(self):
        self.assertEqual(slots.extract("city", "weather in paris please"), "Paris")
        self.assertEqual(slots.extract("city", "tbilisi", expecting=True), "Tbilisi")
        self.assertIsNone(slots.extract("city", "weather"))

    def test_dates(self):
        self.assertEqual(slots.extract("date", "tomorrow", now=NOW), "2026-11-03")
        self.assertEqual(slots.extract("date", "on friday", now=NOW), "2026-11-06")

    def test_time_and_name(self):
        self.assertEqual(slots.extract("time", "at 5:30 pm"), "17:30")
        self.assertEqual(slots.extract("name", "call me alex"), "Alex")

    def test_todo_item(self):
        self.assertEqual(slots.extract("item", "add buy milk to my todo list"), "buy milk")
        self.assertIsNone(slots.extract("item", "add a task"))


class GradientCheck(unittest.TestCase):
    def test_backprop_matches_numerical_gradient(self):
        rng = np.random.default_rng(1)
        net = NumpyMLP(6, 5, 3, dropout=0.0, seed=1)
        X, y = rng.standard_normal((4, 6)), np.array([0, 1, 2, 1])
        _, grads = net.loss_and_grads(X, y)
        h = 1e-6
        for name in ("W0", "b1", "W2"):
            p = net.params[name]
            idx = tuple(0 for _ in p.shape)
            old = p[idx]
            p[idx] = old + h
            up, _ = net.loss_and_grads(X, y)
            p[idx] = old - h
            down, _ = net.loss_and_grads(X, y)
            p[idx] = old
            self.assertAlmostEqual(grads[name][idx], (up - down) / (2 * h), places=5)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = load_intents("intents.json")
        cls.clf = IntentClassifier.fit(samples_from_intents(cls.doc), [], tags_from_intents(cls.doc),
                                       backend="numpy", epochs=100, log=None)
        cls.kb = KnowledgeBase("knowledge")

    def bot(self, **kw):
        log = UnknownLog(os.path.join(tempfile.mkdtemp(), "u.jsonl"))
        return DialogManager(self.clf, self.doc, knowledge=self.kb, unknown_log=log,
                             seed=1, now_fn=lambda: NOW, **kw), Session()

    def say(self, dm, s, text):
        return dm.reply(text, s)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self.clf.save(d)
            loaded = IntentClassifier.load(d)
            a, b = self.clf.predict("hello"), loaded.predict("hello")
            self.assertEqual(a[0][0], b[0][0])
            self.assertAlmostEqual(a[0][1], b[0][1], places=6)

    def test_training_sentences_classified(self):
        self.assertGreater(self.clf.accuracy(samples_from_intents(self.doc)), 0.95)

    def test_slot_filling_and_memory(self):
        dm, s = self.bot()
        self.assertEqual(self.say(dm, s, "weather").text, "Which city?")
        r = self.say(dm, s, "Tbilisi")
        self.assertIn("Tbilisi today", r.text)
        r = self.say(dm, s, "and tomorrow")
        self.assertIn("Tbilisi tomorrow", r.text)  # city carried over

    def test_cancel_pending(self):
        dm, s = self.bot()
        self.say(dm, s, "weather")
        self.assertIn("cancel", self.say(dm, s, "never mind").text.lower())
        self.assertIsNone(s.pending)

    def test_calculator(self):
        dm, s = self.bot()
        self.assertEqual(self.say(dm, s, "what is 17 + 25").text, "17 + 25 = 42")

    def test_context_gates_followups(self):
        dm, s = self.bot()
        self.assertEqual(self.say(dm, s, "another one").route, "clarify")  # no joke yet -> hint
        self.say(dm, s, "tell me a joke")
        self.assertEqual(self.say(dm, s, "another one").intent, "joke_more")

    def test_name_memory_and_todos(self):
        dm, s = self.bot()
        self.say(dm, s, "my name is Priya")
        self.assertIn("Priya", self.say(dm, s, "what's my name").text)
        self.say(dm, s, "add a task")
        self.say(dm, s, "call the dentist")
        self.assertEqual(s.todos, ["call the dentist"])

    def test_knowledge_route(self):
        dm, s = self.bot()
        r = self.say(dm, s, "what is slot filling")
        self.assertEqual(r.route, "knowledge")
        self.assertIn("Slots", r.text)

    def test_unknown_is_logged(self):
        dm, s = self.bot()
        r = self.say(dm, s, "what is the capital of France")
        self.assertEqual(r.route, "fallback")
        self.assertEqual(s.last_unknown, "what is the capital of France")
        self.assertEqual(len(dm.unknown_log.rows()), 1)

    def test_unknown_words_get_uniform_probability(self):
        probs = [p for _, p in self.clf.predict("zzzz qqqq")]
        self.assertAlmostEqual(max(probs), 1 / len(probs), places=6)


@unittest.skipUnless(torch_available(), "PyTorch not installed")
class TorchBackendTests(unittest.TestCase):
    def test_torch_trains_and_roundtrips(self):
        doc = load_intents("intents.json")
        clf = IntentClassifier.fit(samples_from_intents(doc), [], tags_from_intents(doc),
                                   backend="torch", epochs=60, log=None)
        self.assertGreater(clf.accuracy(samples_from_intents(doc)), 0.9)
        with tempfile.TemporaryDirectory() as d:
            clf.save(d)
            self.assertEqual(IntentClassifier.load(d).predict("hello")[0][0], clf.predict("hello")[0][0])


if __name__ == "__main__":
    unittest.main()
