"""Honest evaluation: k-fold cross-validation, confusions and a calibration table.

    python evaluate.py --backend numpy --folds 5

Why this matters: accuracy on sentences the model was trained on tells you nothing.
Here every sentence is predicted by a model that never saw it.  The calibration
table shows whether "90% confident" really means "right 90% of the time" - use it
to choose the --hi / --lo thresholds in chat.py.
"""
import argparse
import random
from collections import Counter

from bot.classifier import IntentClassifier, samples_from_intents, tags_from_intents
from bot.factory import load_intents

OUT_OF_DOMAIN = ["i like turtles", "asdf qwerty", "the mitochondria is the powerhouse of the cell",
                 "purple monkey dishwasher", "can you sing opera", "my cat knocked over a plant"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intents", default="intents.json")
    ap.add_argument("--encoder", default="bow", choices=["bow", "sbert"])
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    doc = load_intents(a.intents)
    samples, tags = samples_from_intents(doc), tags_from_intents(doc)
    random.Random(a.seed).shuffle(samples)
    folds = [samples[i::a.folds] for i in range(a.folds)]

    results = []          # (text, true, pred, confidence)
    last_clf = None
    for k in range(a.folds):
        test = folds[k]
        train = [s for j, f in enumerate(folds) if j != k for s in f]
        clf = IntentClassifier.fit(train, [], tags, encoder=a.encoder, backend=a.backend,
                                   epochs=a.epochs, seed=a.seed, log=None)
        last_clf = clf
        for text, true in test:
            pred, p = clf.predict(text, 1)[0]
            results.append((text, true, pred, p))
        print(f"fold {k + 1}/{a.folds}: {clf.accuracy(test):.1%}")

    acc = sum(t == p for _, t, p, _ in results) / len(results)
    print(f"\ncross-validated accuracy: {acc:.1%} over {len(results)} sentences")

    print("\nmost common mistakes (true -> predicted):")
    for (t, p), n in Counter((t, p) for _, t, p, _ in results if t != p).most_common(8):
        print(f"  {n}x  {t} -> {p}")

    print("\ncalibration (is confidence honest?):")
    print("  confidence   n    actually correct")
    for lo, hi in [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]:
        rows = [r for r in results if lo <= r[3] < hi]
        if rows:
            print(f"  {lo:.2f}-{min(hi, 1):.2f}  {len(rows):4d}   {sum(r[1] == r[2] for r in rows) / len(rows):.0%}")

    print("\nout-of-domain sentences (ideal: low confidence or out_of_scope):")
    for text in OUT_OF_DOMAIN:
        tag, p = last_clf.predict(text, 1)[0]
        print(f"  {text!r:55} -> {tag} ({p:.2f})")


if __name__ == "__main__":
    main()
