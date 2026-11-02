# Sam - a chatbot you can learn from

An upgraded version of the basic "intent -> canned reply" PyTorch chatbot. It still
uses a small neural network to classify intents, but wraps it in the parts that make
a bot feel like it is *thinking*: confidence-aware routing, slot filling, memory,
context, tools, document retrieval, and a learn-from-mistakes loop.

## Quick start

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt                      # numpy (+ torch, recommended)
python chat.py                                       # trains automatically on first run
```

No PyTorch? It still works: a from-scratch NumPy network is used automatically
(`--backend numpy` forces it). Other entry points:

| Command | What it does |
|---|---|
| `python chat.py` | terminal chat (`/help` lists commands) |
| `python web.py` | chat UI at http://127.0.0.1:8000 (standard library only) |
| `python train.py` | retrain after editing `intents.json` |
| `python evaluate.py` | honest cross-validated accuracy + calibration table |
| `python -m unittest discover -s tests -v` | run the tests |

## Try these conversations

```
weather                      -> Which city?  ->  Tbilisi  ->  and tomorrow?   (slot filling + memory)
what is 17 + 25 / what is 2 to the power of 10                                  (safe calculator tool)
tell me a joke  ->  another one                                                 (context unlocks follow-ups)
another one  (with no joke before)                                              (context guard)
my name is Priya  ... later ...  what's my name                                 (long-term memory)
add a task  ->  call the dentist  ->  show my todo list                         (free-text slot)
how do you work / what is slot filling                                          (retrieval from knowledge/)
i like turtles  ->  /teach chitchat                                             (learning loop)
/debug                                                                          (see routes + confidences)
```

## How a message flows

```
text -> normalize/tokenize -> encoder -> neural net -> probability per intent
                                                          |
   +------------------------------------------------------+
   v
 pending question? --yes--> fill the slot, ask next / finish
   | no
   v
 confidence >= 0.75 ........ run intent (reply, action/tool, or ask for missing slots)
 knowledge match >= 0.35 ... answer from knowledge/ (RAG if an LLM is enabled)
 confidence >= 0.30 ........ "Did you mean X?"  (yes -> run it, no -> rephrase)
 otherwise ................. LLM (if enabled) or honest "I don't know"  + logged
```

## Project map (and what each file teaches)

| File | Concept |
|---|---|
| `bot/text.py` | normalisation, stemming; why numbers/symbols become tokens |
| `bot/encoders.py` | bag-of-words vs sentence embeddings behind one interface |
| `bot/models.py` | the same MLP in **PyTorch** and in **pure NumPy** (manual backprop + Adam) |
| `bot/classifier.py` | augmentation, validation split, early stopping, out-of-vocabulary guard |
| `bot/slots.py` | entity extraction with regex, ISO dates, "expecting" mode |
| `bot/actions.py` | tool registry; AST-based safe calculator instead of `eval()` |
| `bot/dialog.py` | confidence routing, clarification, slot state machine, context |
| `bot/knowledge.py` | TF-IDF retrieval (the R in RAG) |
| `bot/llm.py` | optional hybrid fallback to Ollama or Anthropic via `urllib` |
| `bot/memory.py` | session state, persistence, unknown-message log |
| `evaluate.py` | cross-validation, confusion pairs, reliability (calibration) table |
| `tests/test_bot.py` | unit tests incl. a numerical **gradient check** |

## Adding things

* **New intent**: add `tag`, `patterns`, `responses` to `intents.json`, then `python train.py`.
  Aim for 10+ varied example sentences per intent.
* **New tool**: write `@action("name")` in `bot/actions.py`, reference it with `"action": "name"`.
* **New slot**: `"slots": {"city": {"type": "city", "prompt": "Which city?", "remember": true}}`.
  Types: `city date time number name item`.
* **New knowledge**: drop `.md` / `.txt` files in `knowledge/` (paragraphs are the unit).
* **Teach live**: ask something it fails on, then `/teach <intent>` in the terminal.

## Optional upgrades

```bash
# 1. Semantic understanding (paraphrases it never saw)
pip install sentence-transformers
python train.py --encoder sbert && python chat.py --encoder sbert

# 2. Hybrid with a local LLM (free): install Ollama, `ollama pull llama3.2`
BOT_LLM=ollama python chat.py

# 3. Hybrid with a hosted model
export ANTHROPIC_API_KEY=...   # optional: ANTHROPIC_MODEL=<model name>
BOT_LLM=anthropic python chat.py
```
With an LLM enabled, unknown questions go to the model with recent chat history, and
questions that match `knowledge/` are answered *from those passages* (RAG).

## Learning exercises (easy -> hard)

1. Run `evaluate.py`. Which two intents get confused most? Add 5 sentences to the weaker one and re-run.
2. Change `hi`/`lo` in `chat.py` using the calibration table. What do you gain and lose?
3. Add a `time` slot and an intent "set a reminder for 5pm" (the extractor already exists).
4. Add a `clear my list` intent and a `todo_clear` action.
5. Replace the stemmer in `text.py` with NLTK's `PorterStemmer`; compare cross-validated accuracy.
6. Compare `--encoder bow` and `--encoder sbert` on paraphrases like "will I need a brolly?".
7. Make the OOD guard smarter: reject predictions when the top two intents are within 0.1.
8. Replace the demo `weather` action with a real API (Open-Meteo needs no key; use `urllib`).
9. Add temperature scaling (fit one number on validation logits) to calibrate probabilities.
10. Swap TF-IDF retrieval in `knowledge.py` for embeddings and measure which finds the right paragraph.
11. Add a `sentiment` slot-less intent that changes the reply tone, or a second language.
12. Implement the refit trick yourself: pick the epoch count from validation, then retrain on everything
    (`factory.train_and_save` already does a simple version).

## Honest limitations

* Cross-validated accuracy of the bag-of-words model is roughly 70% on the bundled data (see
  `evaluate.py`): it only understands words it has seen. Embeddings, more examples, and the
  out-of-scope intent are the levers. Do not trust a "99% accuracy" measured on training data.
* Regex slot extraction is brittle; weather is demo data; the LLM, sentence-transformer and PyTorch paths
  need their packages/network and are not covered by the offline unit tests (the PyTorch test runs
  automatically when torch is installed).
