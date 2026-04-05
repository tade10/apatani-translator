# Apatani Translator — Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        USER (Browser)                       │
│              types text, picks direction                    │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP POST /translate
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Web App                             │
│                   (translator_app.py)                       │
│                                                             │
│   - Serves the HTML page                                    │
│   - Receives text + direction (en→ap or ap→en)              │
│   - Calls the Translation Engine                            │
│   - Returns JSON result to browser                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Translation Engine                         │
│                   (translator.py)                           │
│                                                             │
│   Step 1: Is the input a single word?                       │
│           │                                                 │
│           ├─ YES → Dictionary Lookup                        │
│           │        ├─ Exact match in CSV                    │
│           │        ├─ Fuzzy match (close spelling)          │
│           │        └─ Not found → "unknown word"            │
│           │                                                 │
│           └─ NO (phrase/sentence) → ML Model               │
│                    ├─ LSTM Seq2Seq Model (trained)          │
│                    └─ Fallback: word-by-word lookup         │
└──────────┬──────────────────────────┬───────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐      ┌───────────────────────────┐
│  Dictionary Data │      │      ML Model Files        │
│                  │      │                            │
│ apatani_words    │      │  model.keras               │
│   .csv           │      │  tokenizer_ap.pkl          │
│ (8,971 pairs)    │      │  tokenizer_en.pkl          │
│                  │      │  model_metadata.pkl        │
│ english_apatani  │      │                            │
│   _index.csv     │      │  (created by train.py)     │
│ (4,766 pairs)    │      │                            │
└──────────────────┘      └───────────────────────────┘


## File Structure

language translator/
│
├── translator_app.py        ← Flask web server (we build this)
├── translator.py            ← Translation engine (we build this)
├── train.py                 ← ML model training script (we build this)
│
├── apatani_words.csv        ← Dictionary: AP → EN  (8,971 pairs)
├── english_apatani_index.csv← Index:      EN → AP  (4,766 pairs)
├── apatani_sentences.csv    ← Sentence pairs for training (1,016)
│
├── templates/
│   └── index.html           ← The web UI (we build this)
│
├── model.keras              ← Trained model (created after training)
├── tokenizer_ap.pkl         ← Apatani tokenizer (created after training)
├── tokenizer_en.pkl         ← English tokenizer (created after training)
└── model_metadata.pkl       ← Max lengths, vocab sizes (created after training)


## Build Order

1. translator.py       — dictionary lookup (works immediately, no training needed)
2. train.py            — trains the LSTM model on sentence pairs
3. translator.py       — add ML model support on top of dictionary
4. translator_app.py   — Flask app wiring it all together
5. templates/index.html— the UI


## How the LSTM Model Works (simplified)

  "Hello" ──► [Encoder LSTM] ──► context vector ──► [Decoder LSTM] ──► "No ayaswdu lo"
               learns to compress                    learns to generate
               the meaning                           the translation

  The model is trained on our 1,016 sentence pairs.
  It learns patterns like: which Apatani words tend to
  appear when certain English words are used.
```
