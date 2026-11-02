"""Terminal chat.   python chat.py --help

Commands inside the chat:
  /help            show this list
  /debug           toggle debug info (route, confidence, top-3 intents, slots)
  /intents         list what the bot knows
  /teach <intent>  after the bot failed to understand you, add your last message
                   to <intent> (new name = create it) and retrain - the learning loop
  /unknown         most common messages the bot did not understand
  /reset           forget the current conversation (keeps saved name / to-dos)
  /quit            leave
"""
import argparse
import os

from bot.factory import (add_intent, add_pattern, build_dialog, model_exists,
                         train_and_save)
from bot.classifier import IntentClassifier
from bot.memory import Session


def main():
    ap = argparse.ArgumentParser(description="Chat with Sam in the terminal")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--intents", default="intents.json")
    ap.add_argument("--knowledge", default="knowledge")
    ap.add_argument("--session", default="data/session.json", help="long-term memory file")
    ap.add_argument("--encoder", default="bow", choices=["bow", "sbert"])
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"])
    ap.add_argument("--hi", type=float, default=0.75, help="answer directly above this confidence")
    ap.add_argument("--lo", type=float, default=0.30, help="ask 'did you mean' above this confidence")
    ap.add_argument("--no-llm", action="store_true", help="ignore BOT_LLM even if set")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    if not model_exists(a.model_dir):
        print("No trained model found - training one now (takes a few seconds)...")
        train_and_save(a.intents, a.model_dir, a.encoder, a.backend)

    dm = build_dialog(a.model_dir, a.intents, a.knowledge, use_llm=not a.no_llm, hi=a.hi, lo=a.lo)
    session = Session.load(a.session)
    debug = a.debug
    print("Sam: Hi! I'm Sam. Type /help for commands, /quit to leave.")
    if dm.llm:
        print(f"     (LLM fallback enabled: {dm.llm.provider}/{dm.llm.model})")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue

        if text.startswith("/"):
            cmd, _, arg = text[1:].partition(" ")
            arg = arg.strip()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                print(__doc__)
            elif cmd == "debug":
                debug = not debug
                print(f"[debug {'on' if debug else 'off'}]")
            elif cmd == "intents":
                for tag, intent in dm.intents.items():
                    print(f"  {tag:14} {len(intent.get('patterns', [])):3} examples  - {intent.get('description', '')}")
            elif cmd == "unknown":
                rows = dm.unknown_log.most_common() if dm.unknown_log else []
                print("\n".join(f"  {n}x  {t}" for t, n in rows) if rows else "  (nothing logged yet)")
            elif cmd == "reset":
                session.reset_short_term()
                print("[conversation reset]")
            elif cmd == "teach":
                if not arg:
                    print("Usage: /teach <intent>   (see /intents for names)")
                elif not session.last_unknown:
                    print("Nothing to teach yet - say something the bot doesn't understand first.")
                else:
                    sentence = session.last_unknown
                    if add_pattern(a.intents, arg, sentence):
                        print(f"[added \"{sentence}\" to '{arg}']")
                    else:
                        resp = input(f"'{arg}' is a new intent. What should I reply to \"{sentence}\"? ").strip()
                        if not resp:
                            print("[cancelled]")
                            continue
                        add_intent(a.intents, arg, sentence, resp)
                        print(f"[created intent '{arg}']")
                    print("[retraining...]")
                    train_and_save(a.intents, a.model_dir, a.encoder, a.backend, log=None)
                    dm.classifier = IntentClassifier.load(a.model_dir)
                    from bot.factory import load_intents
                    dm.set_intents(load_intents(a.intents))
                    session.last_unknown = None
                    print("[done - try your sentence again]")
            else:
                print("Unknown command. Type /help.")
            continue

        reply = dm.reply(text, session)
        print(f"Sam: {reply.text}")
        if debug:
            top = ", ".join(f"{t} {p:.2f}" for t, p in reply.debug.get("top", []))
            print(f"     [route={reply.route} intent={reply.intent} conf={reply.confidence:.2f} "
                  f"context={session.context}] top: {top}")
            if reply.debug.get("slots"):
                print(f"     slots: {reply.debug['slots']}")
        session.save(a.session)
    session.save(a.session)
    print("Sam: Bye!")


if __name__ == "__main__":
    main()
