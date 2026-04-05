"""
Extracts word pairs and sentence pairs from Dictionary-of-the-Apatani-Language.pdf.

Two sections:
  - Pages 14-394  (PDF pages): Apatani-English dictionary
  - Pages 395-713 (PDF pages): English-Apatani index
"""

import re
import csv
import pdfplumber

PDF_PATH = "Dictionary-of-the-Apatani-Language.pdf"

# PDF page indices (0-based)
AP_EN_START = 13   # PDF page 14
AP_EN_END   = 394  # PDF page 394 (inclusive)
EN_AP_START = 394  # PDF page 395
EN_AP_END   = 712  # PDF page 713 (inclusive)

POS_TAGS = r'(?:n|v|vi|vt|vr|vi-r|vt-r|vcn|adj|adv|adjphr|advphr|pron|interj|conj|part|suff|vsuff|adjsuff|num|class|prep|exp|dem|dl)\b[:\-\w]*\.?'


# ── English-Apatani Index ────────────────────────────────────────────────────

def parse_en_ap_line(line):
    """
    Parse one line of the English-Apatani index.
    Format: english_phrase [qualifier] pos. apatani_word(s)
    Returns (english, apatani) or None.
    """
    line = line.strip()
    if not line:
        return None
    # Skip page-number-only lines and section headers
    if re.match(r'^\d+$', line):
        return None
    if re.match(r'^[A-Z]-[a-z]$', line):
        return None

    # Match: text  pos_tag  apatani_words
    # pos tags end with a period or colon
    m = re.match(
        r'^(.+?)\s+(?:' + POS_TAGS + r')\s+(.+)$',
        line
    )
    if m:
        english = m.group(1).strip().rstrip('(').strip()
        apatani = m.group(2).strip()
        # Remove dialect/usage notes like (Hi)/(Bu) at the end
        apatani = re.sub(r'\s*\([A-Z][a-z]*(?:,\s*[A-Z][a-z]*)*\)\s*$', '', apatani).strip()
        # Skip if either side is empty or looks like a page number
        if not english or not apatani:
            return None
        if re.match(r'^\d+$', apatani):
            return None
        return (english.lower(), apatani)
    return None


def extract_en_ap(pdf):
    pairs = []
    for page_idx in range(EN_AP_START, EN_AP_END + 1):
        text = pdf.pages[page_idx].extract_text()
        if not text:
            continue
        for line in text.splitlines():
            result = parse_en_ap_line(line)
            if result:
                pairs.append(result)
    return pairs


# ── Apatani-English Dictionary ───────────────────────────────────────────────

def extract_ap_en(pdf):
    """
    Extract word pairs and sentence pairs from the Apatani-English section.
    Word pair:     Apatani headword → primary English definition
    Sentence pair: Apatani example  → English translation
    """
    # Collect all page text into one block, then split into entries
    all_text = []
    for page_idx in range(AP_EN_START, AP_EN_END + 1):
        text = pdf.pages[page_idx].extract_text()
        if text:
            # Remove page number lines (lone numbers / section headers like "A-a")
            lines = [l for l in text.splitlines()
                     if not re.match(r'^\s*\d+\s*$', l)
                     and not re.match(r'^[A-Z]-[a-z]$', l.strip())]
            all_text.append(' '.join(lines))

    full_text = ' '.join(all_text)
    # Collapse multiple spaces
    full_text = re.sub(r'  +', ' ', full_text)

    word_pairs = []
    sentence_pairs = []

    # Split into entries: each entry starts with a lowercase word possibly
    # followed by optional pronunciation in [...] and a POS tag.
    # We use a lookahead to split on entry boundaries.
    entry_pattern = re.compile(
        r'(?<!\w)([a-z][a-záàâãäåæçèéêëìíîïðñòóôõöùúûüýþ\'\-]*'
        r'(?:\s[a-záàâãäåæçèéêëìíîïðñòóôõöùúûüýþ\'\-]+)?)'  # headword (1-2 words)
        r'(?:\s*\[[^\]]*\])?'                                  # optional [pronunciation]
        r'\s+(?:' + POS_TAGS + r')'                           # POS tag
        r'\s+(\d+\.\s+)?'                                     # optional "1. "
    )

    # Find all entry start positions
    starts = [(m.start(), m.group(1).strip()) for m in entry_pattern.finditer(full_text)]

    for i, (start, headword) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(full_text)
        entry_text = full_text[start:end].strip()

        # ── Word pair ──
        # Primary definition: text after pos tag, before first example sentence
        # or before "Syn:" / "Ant:" / "Encycl:" / "Var:" / numbered sense "2."
        def_match = re.match(
            r'.+?(?:' + POS_TAGS + r')\s+(?:\d+\.\s+)?'  # skip headword + pos
            r'((?:(?!\d+\.|Syn:|Ant:|Encycl:|Var:|Gramm:|Usage:|Etym:)[^.!?])+[.!?]?)',
            entry_text
        )
        if def_match:
            raw_def = def_match.group(1).strip().rstrip('.')
            # Skip if definition looks like a proper-noun-only or code
            if raw_def and len(raw_def) > 2 and not re.match(r'^[\d\*\[\]]+$', raw_def):
                # Trim to first semicolon-separated synonym list if very long
                word_pairs.append((headword, raw_def))

        # ── Sentence pairs ──
        # Pattern: a sentence ending with punctuation immediately followed by
        # an English translation (starts with capital letter, ends with punctuation)
        # This is hard to do perfectly; we look for pairs like:
        # "Apatani sentence[.!?]  English sentence[.!?]"
        # Heuristic: Apatani sentences rarely start with common English words
        ENGLISH_STARTERS = re.compile(
            r'^(I |He |She |They |We |You |It |The |A |An |This |That |There |'
            r'Has |Have |Did |Do |Does |Is |Are |Was |Were |When |Where |'
            r'What |Who |How |Please |Don\'t |Let )',
            re.IGNORECASE
        )
        sentences = re.findall(r'([A-Z][^.!?]{5,80}[.!?])', entry_text)
        for j in range(len(sentences) - 1):
            s1 = sentences[j].strip()
            s2 = sentences[j + 1].strip()
            # s1 should be Apatani (no common English starter)
            # s2 should be English (has common English starter)
            if not ENGLISH_STARTERS.match(s1) and ENGLISH_STARTERS.match(s2):
                sentence_pairs.append((s1, s2))

    return word_pairs, sentence_pairs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Reading PDF...")
    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"  Total pages: {len(pdf.pages)}")

        print("Extracting Apatani-English word/sentence pairs...")
        ap_en_words, ap_en_sentences = extract_ap_en(pdf)
        print(f"  Word pairs:    {len(ap_en_words)}")
        print(f"  Sentence pairs:{len(ap_en_sentences)}")

        print("Extracting English-Apatani index...")
        en_ap_pairs = extract_en_ap(pdf)
        print(f"  EN→AP pairs:   {len(en_ap_pairs)}")

    # ── Save Apatani-English words (merge with scraped website data) ──
    existing_words = set()
    try:
        with open('apatani_words.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            existing_words = {(r[0].strip(), r[1].strip()) for r in reader if r}
    except FileNotFoundError:
        pass

    all_words = list(existing_words)
    seen_ap = {p[0].lower() for p in existing_words}
    for ap, en in ap_en_words:
        if ap.lower() not in seen_ap:
            all_words.append((ap, en))
            seen_ap.add(ap.lower())

    with open('apatani_words.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Apatani', 'English'])
        writer.writerows(sorted(all_words, key=lambda x: x[0].lower()))
    print(f"\napatani_words.csv → {len(all_words)} word pairs")

    # ── Save English-Apatani index ──
    with open('english_apatani_index.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['English', 'Apatani'])
        writer.writerows(en_ap_pairs)
    print(f"english_apatani_index.csv → {len(en_ap_pairs)} pairs")

    # ── Save sentence pairs (merge everything) ──
    existing_sentences = []
    try:
        with open('apatani_sentences.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            existing_sentences = [(r[0].strip(), r[1].strip()) for r in reader if r]
    except FileNotFoundError:
        pass

    seen_sent = {s[0].lower() for s in existing_sentences}
    all_sentences = list(existing_sentences)
    for ap, en in ap_en_sentences:
        if ap.lower() not in seen_sent:
            all_sentences.append((ap, en))
            seen_sent.add(ap.lower())

    with open('apatani_sentences.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Apatani', 'English'])
        writer.writerows(all_sentences)
    print(f"apatani_sentences.csv → {len(all_sentences)} sentence pairs")

    # ── Preview samples ──
    print("\n--- Sample EN→AP index ---")
    for en, ap in en_ap_pairs[:10]:
        print(f"  {en:<30} → {ap}")

    print("\n--- Sample AP→EN words (from PDF) ---")
    for ap, en in ap_en_words[:10]:
        print(f"  {ap:<20} → {en}")

    print("\n--- Sample sentence pairs (from PDF) ---")
    for ap, en in ap_en_sentences[:10]:
        print(f"  AP: {ap}")
        print(f"  EN: {en}")
        print()


if __name__ == '__main__':
    main()
