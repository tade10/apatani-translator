import csv
import difflib
import os
import pickle
import re

# ── Data Loading ─────────────────────────────────────────────────────────────
#
# We load both CSVs into plain Python dicts.
# A dict gives us O(1) lookup — looking up any word is instant
# regardless of how large the dictionary gets.
#
# ap_to_en: { "áamì": "cat", "aarda": "tomorrow", ... }
# en_to_ap: { "cat": "áamì", "tomorrow": "aarda", ... }

def load_dictionaries():
    ap_to_en = {}  # Apatani word  → English meaning
    en_to_ap = {}  # English word  → Apatani word

    # Load Apatani→English word pairs
    with open('apatani_words.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ap = row['Apatani'].strip()
            en = row['English'].strip()
            if ap and en:
                # Store lowercase key so "Cat" and "cat" both match
                ap_to_en[ap.lower()] = en

    # Load English→Apatani index
    with open('english_apatani_index.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            en = row['English'].strip()
            ap = row['Apatani'].strip()
            if en and ap:
                en_to_ap[en.lower()] = ap

    return ap_to_en, en_to_ap


# Load once when this file is imported — not on every translation request
ap_to_en, en_to_ap = load_dictionaries()

print(f"Dictionary loaded: {len(ap_to_en)} AP→EN entries, {len(en_to_ap)} EN→AP entries")


# ── ML Model Loading ──────────────────────────────────────────────────────────
#
# We only load the model if the files exist (i.e. training has been run).
# This way the app still works using dictionary-only mode if model is missing.
#
# We also build reverse index maps: index → word
# During training, tokenizers map words → numbers.
# At inference we need the reverse: numbers → words, to read the output.

def load_ml_model():
    if not os.path.exists('model.keras'):
        print("No trained model found — using dictionary-only mode.")
        return None, None, None, None

    from tensorflow.keras.models import load_model
    model = load_model('model.keras')

    with open('tokenizer_ap.pkl', 'rb') as f:
        tokenizer_ap = pickle.load(f)
    with open('tokenizer_en.pkl', 'rb') as f:
        tokenizer_en = pickle.load(f)
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"ML model loaded — vocab AP:{metadata['vocab_size_ap']} EN:{metadata['vocab_size_en']}")
    return model, tokenizer_ap, tokenizer_en, metadata

model, tokenizer_ap, tokenizer_en, metadata = load_ml_model()


# ── Dictionary Lookup ─────────────────────────────────────────────────────────
#
# lookup_word tries two strategies in order:
#   1. Exact match  — "cat" → "áamì"  (fast, perfect)
#   2. Fuzzy match  — "tommorrow" finds "tomorrow" (handles typos)
#
# cutoff=0.6 means the match must be at least 60% similar.
# Lower = more lenient (more false matches), Higher = more strict.

def lookup_word(word, direction='en_to_ap'):
    key = word.strip().lower()
    dictionary = en_to_ap if direction == 'en_to_ap' else ap_to_en

    # 1. Exact match
    if key in dictionary:
        return dictionary[key], 'exact'

    # 2. Fuzzy match
    # get_close_matches returns a list of the closest keys, best match first
    matches = difflib.get_close_matches(key, dictionary.keys(), n=1, cutoff=0.6)
    if matches:
        best = matches[0]
        return dictionary[best], f'fuzzy (matched "{best}")'

    return None, 'not found'


# ── Sentence Translation ──────────────────────────────────────────────────────
#
# translate_text splits the input into individual words and looks up each one.
#
# For example: "the cat sleeps" → ["the", "cat", "sleeps"]
#   "the"    → not found  → kept as "[the]"
#   "cat"    → áamì
#   "sleeps" → not found  → kept as "[sleeps]"
# Result: "[the] áamì [sleeps]"
#
# Words wrapped in [brackets] signal to the user they weren't translated.
# This is honest — better than silently dropping words.

# ── ML Inference ─────────────────────────────────────────────────────────────
#
# ml_translate uses the trained LSTM to translate a full sentence.
#
# How greedy decoding works step by step:
#
#   1. Encode input:  "I want water" → [4, 12, 7, 0, 0] (padded sequence)
#   2. Run encoder:   sequence → context vector (hidden state)
#   3. Decode loop:
#        feed <start> token → model predicts most likely first word
#        feed that word back → model predicts second word
#        ... repeat until <end> token or max length reached
#   4. Convert predicted indices back to words

def ml_translate(text, direction='ap_to_en'):
    if model is None:
        return None  # no model trained yet, fall back to dictionary

    # The model was trained on Apatani→English only.
    # EN→AP direction falls back to word-by-word dictionary lookup.
    if direction != 'ap_to_en':
        return None

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np

    # Step 1: tokenize and pad the input
    seq = tokenizer_ap.texts_to_sequences([text.lower()])
    enc_input = pad_sequences(seq, maxlen=metadata['max_ap_len'], padding='post')

    # Step 2: build index→word map for English (reverse of word→index)
    index_to_word = {idx: word for word, idx in tokenizer_en.word_index.items()}
    # The Keras tokenizer strips angle brackets, so <start>→'start', <end>→'end'
    start_token = tokenizer_en.word_index.get('start')
    end_token   = tokenizer_en.word_index.get('end')

    # Step 3: one-shot decoding
    # Feed [start, 0, 0, ...] as decoder input — the model predicts all positions at once.
    # Then read the output left to right, stopping at the 'end' token.
    dec_input   = pad_sequences([[start_token]], maxlen=metadata['max_en_len'] - 1, padding='post')
    predictions = model.predict([enc_input, dec_input], verbose=0)  # shape: (1, seq_len, vocab)

    decoded_words = []
    for step in range(predictions.shape[1]):
        next_index = int(np.argmax(predictions[0, step, :]))
        next_word  = index_to_word.get(next_index, '')

        if next_index == end_token or not next_word:
            break
        if next_word not in ('start', 'end', '<OOV>'):
            decoded_words.append(next_word)

    return ' '.join(decoded_words) if decoded_words else None


# ── Sentence Translation ──────────────────────────────────────────────────────
#
# translate_text is the main entry point.
# It tries strategies in order, best first:
#   1. ML model      — handles full sentences naturally (AP→EN only for now)
#   2. Word-by-word  — fallback using dictionary lookup per word

def translate_text(text, direction='en_to_ap'):
    # Strategy 1: try the ML model (works best for full sentences, AP→EN only)
    ml_result = ml_translate(text, direction)
    if ml_result:
        return ml_result, 'ml'

    # Strategy 2: word-by-word dictionary lookup (works for both directions)
    words = re.findall(r"[\w']+|[^\w\s]", text)
    translated = []
    for word in words:
        # Pass punctuation through unchanged
        if not any(c.isalpha() for c in word):
            translated.append(word)
            continue

        result, _ = lookup_word(word, direction)
        translated.append(result if result else f'[{word}]')

    return ' '.join(translated), 'dictionary'
