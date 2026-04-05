import requests
import csv
import re
import time
from bs4 import BeautifulSoup

BASE_URL = "https://language.apatani.org/dictionary/{}.html"
LETTERS = "abcdefghijklmnopqrstuvwxyz"

def parse_entry(p_tag):
    """Parse a single <p> entry into word pairs and sentence pairs."""
    word_pairs = []
    sentence_pairs = []

    # Get the Apatani headword from <strong>
    strong = p_tag.find('strong')
    if not strong:
        return word_pairs, sentence_pairs
    apatani_word = strong.get_text().strip()

    # Get full text of the paragraph
    full_text = p_tag.get_text()

    # Remove the word and pronunciation /.../ from the front
    text = full_text
    text = text.replace(apatani_word, '', 1).strip()
    text = re.sub(r'^/[^/]+/\s*', '', text).strip()

    # Remove part-of-speech tag at the start (n. v. adj. adv. vr. int. pron. conj. etc.)
    text = re.sub(r'^(?:n\.|v\.|vr\.|adj\.|adv\.|pron\.|int\.|conj\.|prep\.)\s*', '', text).strip()

    # Remove variant/synonym/antonym/origin notes like [var. xxx] [syn. xxx]
    text = re.sub(r'\[(?:var\.|syn\.|ant\.|or\.|adv\.)[^\]]*\]', '', text).strip()

    # Extract English definition: text up to the first example sentence
    # Example sentences appear as <em> tags in the HTML; in plain text they
    # are followed by a [bracketed English translation]
    # Strategy: split on the pattern "Apatani sentence [English]"
    sentence_pattern = re.compile(r'([^.\[!?]*[.!?])\s*\[([^\]]+)\]')

    # Get the definition: everything before the first sentence example
    first_example = sentence_pattern.search(text)
    if first_example:
        raw_def = text[:first_example.start()].strip().rstrip('.')
    else:
        raw_def = text.strip().rstrip('.')

    # Clean definition: take first meaning (before semicolons is fine, keep all)
    english_def = raw_def.strip()
    if english_def:
        word_pairs.append((apatani_word, english_def))

    # Extract all sentence pairs
    for match in sentence_pattern.finditer(text):
        ap_sentence = match.group(1).strip()
        en_sentence = match.group(2).strip()
        # Skip meta notes like [var. háagyá]
        if re.match(r'^(var\.|syn\.|ant\.|or\.|adv\.)', en_sentence):
            continue
        if len(ap_sentence) > 4 and len(en_sentence) > 4:
            sentence_pairs.append((ap_sentence, en_sentence))

    return word_pairs, sentence_pairs


def scrape_letter(letter):
    url = BASE_URL.format(letter)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return [], []

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Dictionary entries are <p> tags inside the article
    article = soup.find('article')
    if not article:
        # Fallback: search all <p> tags with a <strong> child
        paragraphs = [p for p in soup.find_all('p') if p.find('strong')]
    else:
        paragraphs = [p for p in article.find_all('p') if p.find('strong')]

    word_pairs = []
    sentence_pairs = []
    for p in paragraphs:
        wp, sp = parse_entry(p)
        word_pairs.extend(wp)
        sentence_pairs.extend(sp)

    print(f"  [{letter.upper()}] {len(paragraphs)} entries → {len(word_pairs)} word pairs, {len(sentence_pairs)} sentence pairs")
    return word_pairs, sentence_pairs


def main():
    all_word_pairs = []
    all_sentence_pairs = []

    print("Scraping Apatani dictionary (a-z)...")
    for letter in LETTERS:
        wp, sp = scrape_letter(letter)
        all_word_pairs.extend(wp)
        all_sentence_pairs.extend(sp)
        time.sleep(0.5)  # be polite to the server

    # Save word pairs
    with open('apatani_words.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Apatani', 'English'])
        writer.writerows(all_word_pairs)

    # Save sentence pairs (merge with existing apatani_data.csv)
    existing = set()
    try:
        with open('apatani_data.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if row:
                    existing.add((row[0].strip(), row[1].strip()))
                    all_sentence_pairs.append((row[0].strip(), row[1].strip()))
    except FileNotFoundError:
        pass

    # Deduplicate sentence pairs
    seen = set()
    unique_sentences = []
    for pair in all_sentence_pairs:
        key = pair[0].lower()
        if key not in seen:
            seen.add(key)
            unique_sentences.append(pair)

    with open('apatani_sentences.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Apatani', 'English'])
        writer.writerows(unique_sentences)

    print(f"\nDone!")
    print(f"  apatani_words.csv     → {len(all_word_pairs)} word pairs")
    print(f"  apatani_sentences.csv → {len(unique_sentences)} sentence pairs")


if __name__ == '__main__':
    main()
