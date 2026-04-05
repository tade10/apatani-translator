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

def clean_definition(text):
    """Trim a verbose definition to its first short meaning.
    'father; daddy; papa' → 'father'
    'to go or come home; to visit someone' → 'to go or come home'
    '[a´pi] n' → None  (parsing artifact, discard)
    """
    if not text:
        return None
    # Discard PDF parsing artifacts that start with [ or contain only symbols
    if text.startswith('[') or re.match(r'^[\[\]/\s\d\'\`\´]+$', text):
        return None
    # Take only the first meaning (before first semicolon)
    first = text.split(';')[0].strip().rstrip('.')
    # Discard if too short or looks like a tag
    if len(first) < 2:
        return None
    return first


def load_dictionaries():
    ap_to_en = {}  # Apatani word  → English meaning
    en_to_ap = {}  # English word  → Apatani word

    # Load Apatani→English word pairs
    with open('apatani_words.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ap = row['Apatani'].strip()
            en = clean_definition(row['English'].strip())
            if ap and en:
                ap_to_en[ap.lower()] = en

    # Load English→Apatani index
    with open('english_apatani_index.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            en = row['English'].strip()
            ap = row['Apatani'].strip()
            if en and ap:
                en_to_ap[en.lower()] = ap

    return ap_to_en, en_to_ap


# ── Common word supplements ───────────────────────────────────────────────────
#
# The academic EN→AP index misses basic conversational words.
# These are handcrafted from the sentence pairs and dictionary examples.
# Add more here as needed — format: 'english': 'apatani'

EN_AP_SUPPLEMENT = {
    # Pronouns
    'i': 'ngo', 'me': 'ngo', 'my': 'ngiika', 'mine': 'ngiika',
    'you': 'no', 'your': 'nwka',
    'he': 'mo', 'she': 'mo', 'his': 'moka', 'her': 'moka',
    'we': 'ngiinyi', 'our': 'ngiinyika',
    'they': 'henku', 'their': 'henkuka',
    'this': 'si', 'that': 'hwka',
    # Common verbs / copula
    'is': 'do', 'am': 'do', 'are': 'do', 'was': 'du', 'were': 'du',
    'have': 'ahi', 'has': 'ahi',
    'eat': 'dii', 'drink': 'tu',
    'go': 'ich', 'come': 'a',
    'want': 'dwnado', 'need': 'dwnado',
    'speak': 'lu', 'say': 'lu', 'tell': 'lu',
    'know': 'helo',
    # Common nouns
    'name': 'tabu', 'person': 'ako', 'people': 'mwlañja',
    'house': 'ude', 'village': 'lemba',
    'water': 'yasi', 'rice': 'apin', 'food': 'apin',
    'father': 'aba', 'mother': 'ane', 'brother': 'aban', 'sister': 'ani',
    'child': 'wqa', 'baby': 'wqa',
    'day': 'lo', 'night': 'ayo', 'morning': 'aro', 'tomorrow': 'arda',
    'year': 'rañ', 'time': 'myodu',
    'sun': 'abo', 'fire': 'mii', 'water': 'yasi',
    # Question words
    'what': 'nii', 'where': 'nona', 'when': 'hendo',
    'who': 'nunu', 'how': 'heni', 'why': 'helo',
    # Common adjectives
    'good': 'aro', 'bad': 'abu', 'big': 'kaye', 'small': 'pichi',
    'many': 'abu', 'much': 'abu', 'little': 'iche',
    'fast': 'aare', 'slow': 'ado',
    'new': 'kanyan', 'old': 'ako',
    # Misc
    'yes': 'ao', 'no': 'ale', 'not': 'ma',
    'here': 'so', 'there': 'hwka',
    'also': 'kw', 'and': 'pe', 'but': 'ke',
    'please': 'pe', 'thank': 'paya', 'welcome': 'hangw',
}

AP_EN_SUPPLEMENT = {v: k for k, v in EN_AP_SUPPLEMENT.items()}


# Load once when this file is imported — not on every translation request
ap_to_en, en_to_ap = load_dictionaries()

# Merge supplements (don't overwrite existing entries — dictionary takes priority)
for k, v in EN_AP_SUPPLEMENT.items():
    if k not in en_to_ap:
        en_to_ap[k] = v
for k, v in AP_EN_SUPPLEMENT.items():
    if k not in ap_to_en:
        ap_to_en[k] = v

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
    original = word.strip()
    dictionary = en_to_ap if direction == 'en_to_ap' else ap_to_en

    # 1. Exact match
    if key in dictionary:
        return dictionary[key], 'exact'

    # 2. Skip fuzzy matching for likely proper nouns (capitalized, length > 2)
    #    and for very short words — too easy to get a wrong fuzzy match
    if (original[0].isupper() and len(original) > 2) or len(original) <= 2:
        return None, 'not found'

    # 3. Fuzzy match — raised cutoff to 0.75 to reduce wrong matches
    matches = difflib.get_close_matches(key, dictionary.keys(), n=1, cutoff=0.75)
    if matches:
        best = matches[0]
        return dictionary[best], f'fuzzy'

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
    # Word-by-word dictionary lookup for both directions.
    # The ML model (AP→EN) is kept for future improvement but disabled for now —
    # its validation accuracy (35%) is too low to produce reliable output.
    words = re.findall(r"[\w']+|[^\w\s]", text)
    translated = []
    for word in words:
        # Pass punctuation through unchanged
        if not any(c.isalpha() for c in word):
            translated.append(word)
            continue

        result, _ = lookup_word(word, direction)
        # Proper nouns that weren't found: pass through as-is (don't bracket names)
        if result is None and word[0].isupper():
            translated.append(word)
        else:
            translated.append(result if result else f'[{word}]')

    return ' '.join(translated), 'dictionary'
