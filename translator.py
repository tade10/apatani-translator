import csv
import difflib

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
