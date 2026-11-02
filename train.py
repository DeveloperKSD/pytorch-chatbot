"""Train the intent classifier.   python train.py --help"""
import argparse

from bot.factory import train_and_save


def main():
    ap = argparse.ArgumentParser(description="Train the intent classifier from intents.json")
    ap.add_argument("--intents", default="intents.json")
    ap.add_argument("--out", default="model", help="directory to save the model")
    ap.add_argument("--encoder", default="bow", choices=["bow", "sbert"],
                    help="bow = bag of words (no downloads); sbert = sentence-transformers embeddings")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"],
                    help="auto = PyTorch if installed, else the from-scratch NumPy network")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val-split", type=float, default=0.2,
                    help="fraction held out for validation/early stopping (0 = train on everything)")
    ap.add_argument("--augment", type=int, default=3, help="augmented copies per sentence")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train_and_save(a.intents, a.out, a.encoder, a.backend, a.hidden, a.epochs, a.lr,
                   a.val_split, a.augment, a.seed)


if __name__ == "__main__":
    main()
