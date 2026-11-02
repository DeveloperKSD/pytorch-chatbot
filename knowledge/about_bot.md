# How Sam works

Here is how Sam works and understands you: every message is turned into numbers, a small neural network scores each intent, and a dialog manager picks the response. Sam understands you by comparing your words with the example sentences it was trained on.

# About this chatbot

Sam is a hybrid chatbot made of several small parts. Every message goes through the same pipeline: the text is normalised and tokenized, an encoder turns it into numbers, a neural network scores every intent, and a dialog manager decides what to do with those scores.

# Intent classification

An intent classifier is a small neural network that reads a sentence and outputs a probability for every intent defined in intents.json, such as greeting, weather or calculator. The network is a feed-forward multilayer perceptron with two hidden layers, ReLU activations and dropout. It is trained on the example patterns you write for each intent.

# Training

Training uses the patterns in intents.json. Each pattern is augmented by randomly dropping words, which multiplies the data and makes the model less dependent on single words. A validation split and early stopping stop training once the model starts memorising instead of generalising. Run python train.py to retrain, or use the teach command inside the chat to add a new example and retrain automatically.

# Confidence thresholds

The dialog manager compares the top probability with two thresholds. Above the high threshold the bot answers directly. Between the low and high thresholds it asks a clarification question such as "did you mean ...?". Below the low threshold it admits it does not know, logs the message to the unknown log, and optionally hands the question to a language model.

# Slot filling

Slots are the details an intent needs, for example the city and date for a weather request. If a required slot is missing the bot asks a follow-up question and keeps the half-finished task in memory until the user answers. Some slots are remembered across turns, so after asking about weather in Paris the follow-up "and tomorrow?" reuses the city.

# Context

Context is short-term memory about the current topic. After telling a joke the bot sets the context to joke, which unlocks follow-up intents like "another one" that would be ambiguous on their own. An intent with a context filter is ignored unless that context is active.

# Memory

Long-term facts such as your name and your to-do list are saved to a JSON session file, so the bot still remembers them after a restart. Short-term state such as pending questions and the active context lives only for the current conversation.

# Retrieval and RAG

This paragraph comes from the knowledge folder. The bot searches every paragraph of the documents in knowledge with TF-IDF similarity and answers from the best match when it is similar enough. Retrieval-augmented generation, or RAG, passes the best passages to a language model so it can answer in its own words while staying grounded in your documents.

# Teaching the bot

To teach the bot something new, type the teach command followed by an intent name right after it failed to understand you. The failed sentence is saved as a new training pattern in intents.json and the model retrains in a few seconds. The unknown command shows the messages the bot struggled with most often, which tells you what to teach next.

# Upgrades

The bag-of-words encoder can be replaced by sentence embeddings so the bot understands paraphrases it never saw. A local or hosted language model can be enabled as a fallback with the BOT_LLM environment variable. Real data sources can be connected by adding functions to actions.py.
