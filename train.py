"""
train.py — trains a seq2seq LSTM model on Apatani-English sentence pairs.

Run this once to produce:
  model.keras           — the trained model
  tokenizer_ap.pkl      — converts Apatani words ↔ numbers
  tokenizer_en.pkl      — converts English words ↔ numbers
  model_metadata.pkl    — max sentence lengths and vocab sizes
"""

import csv
import pickle
import numpy as np

# ── Step 1: Load sentence pairs ───────────────────────────────────────────────
#
# We read every row from the CSV into two parallel lists.
# Index 0 in apatani_sentences matches index 0 in english_sentences, etc.

def load_pairs():
    apatani_sentences = []
    english_sentences = []

    with open('apatani_sentences.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ap = row['Apatani'].strip()
            en = row['English'].strip()
            if ap and en:
                # Add special tokens to English (target language):
                #   <start> tells the decoder "begin generating now"
                #   <end>   tells the decoder "stop generating"
                apatani_sentences.append(ap.lower())
                english_sentences.append(f'<start> {en.lower()} <end>')

    print(f"Loaded {len(apatani_sentences)} sentence pairs")
    return apatani_sentences, english_sentences


# ── Step 2: Tokenization ──────────────────────────────────────────────────────
#
# Neural networks work with numbers, not text.
# A Tokenizer builds a vocabulary: { "cat": 1, "dog": 2, "the": 3, ... }
# Then converts sentences to lists of numbers: "the cat" → [3, 1]
#
# We need two separate tokenizers — one per language.

def build_tokenizers(apatani_sentences, english_sentences):
    from tensorflow.keras.preprocessing.text import Tokenizer

    # oov_token='<OOV>' handles unknown words at inference time
    tokenizer_ap = Tokenizer(oov_token='<OOV>')
    tokenizer_ap.fit_on_texts(apatani_sentences)

    tokenizer_en = Tokenizer(oov_token='<OOV>')
    tokenizer_en.fit_on_texts(english_sentences)

    print(f"Apatani vocabulary size: {len(tokenizer_ap.word_index)}")
    print(f"English vocabulary size:  {len(tokenizer_en.word_index)}")

    return tokenizer_ap, tokenizer_en


# ── Step 3: Prepare training data ─────────────────────────────────────────────
#
# The model needs fixed-length input, so we pad shorter sentences with zeros.
# We also need to split the English side into:
#   decoder_input:  <start> I  want  water
#   decoder_target:         I  want  water <end>
# This teaches the decoder to predict the next word at each step.

def prepare_data(apatani_sentences, english_sentences, tokenizer_ap, tokenizer_en):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Convert sentences → sequences of integers
    ap_seqs = tokenizer_ap.texts_to_sequences(apatani_sentences)
    en_seqs = tokenizer_en.texts_to_sequences(english_sentences)

    # Find the longest sentence in each language (we'll pad everything to this)
    max_ap_len = max(len(s) for s in ap_seqs)
    max_en_len = max(len(s) for s in en_seqs)

    # Pad: shorter sequences get zeros added at the end ('post')
    encoder_input = pad_sequences(ap_seqs, maxlen=max_ap_len, padding='post')

    # decoder_input  = all tokens except the last  (<start> ... word)
    # decoder_target = all tokens except the first (word ... <end>)
    decoder_input  = pad_sequences([s[:-1] for s in en_seqs], maxlen=max_en_len - 1, padding='post')
    decoder_target = pad_sequences([s[1:]  for s in en_seqs], maxlen=max_en_len - 1, padding='post')

    print(f"encoder_input shape:  {encoder_input.shape}  (samples × max_apatani_len)")
    print(f"decoder_input shape:  {decoder_input.shape}  (samples × max_english_len)")
    print(f"decoder_target shape: {decoder_target.shape}")

    return encoder_input, decoder_input, decoder_target, max_ap_len, max_en_len


# ── Step 4: Build the model ───────────────────────────────────────────────────
#
# Architecture:
#
#   Encoder:
#     Embedding → turns word index into a dense vector (64 numbers per word)
#     LSTM      → reads the sequence, outputs a context vector (state)
#
#   Decoder:
#     Embedding → same idea for the target language
#     LSTM      → takes context from encoder, generates output step by step
#     Dense     → converts LSTM output to a probability over the vocabulary

def build_model(vocab_size_ap, vocab_size_en, max_ap_len, max_en_len, embed_dim=64, lstm_units=128):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

    # ── Encoder ──
    encoder_inputs = Input(shape=(max_ap_len,), name='encoder_input')
    enc_embedding  = Embedding(vocab_size_ap, embed_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
    # return_state=True gives us the hidden state (memory) to pass to decoder
    _, enc_h, enc_c = LSTM(lstm_units, return_state=True, name='encoder_lstm')(enc_embedding)
    encoder_states = [enc_h, enc_c]

    # ── Decoder ──
    decoder_inputs = Input(shape=(max_en_len - 1,), name='decoder_input')
    dec_embedding  = Embedding(vocab_size_en, embed_dim, mask_zero=True, name='decoder_embedding')(decoder_inputs)
    # initial_state connects the encoder's memory into the decoder
    dec_lstm_out, _, _ = LSTM(lstm_units, return_sequences=True, return_state=True, name='decoder_lstm')(
        dec_embedding, initial_state=encoder_states
    )
    # Dense layer: at each step, pick the most likely next word
    decoder_outputs = Dense(vocab_size_en, activation='softmax', name='decoder_dense')(dec_lstm_out)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


# ── Step 5: Train and save ────────────────────────────────────────────────────

def main():
    print("=== Loading data ===")
    apatani_sentences, english_sentences = load_pairs()

    print("\n=== Building tokenizers ===")
    tokenizer_ap, tokenizer_en = build_tokenizers(apatani_sentences, english_sentences)

    print("\n=== Preparing training data ===")
    encoder_input, decoder_input, decoder_target, max_ap_len, max_en_len = prepare_data(
        apatani_sentences, english_sentences, tokenizer_ap, tokenizer_en
    )

    vocab_size_ap = len(tokenizer_ap.word_index) + 1  # +1 for padding token (index 0)
    vocab_size_en = len(tokenizer_en.word_index) + 1

    print("\n=== Building model ===")
    model = build_model(vocab_size_ap, vocab_size_en, max_ap_len, max_en_len)

    print("\n=== Training ===")
    # validation_split=0.1 holds back 10% of data to check for overfitting
    # epochs=100 means the model sees the full dataset 100 times
    model.fit(
        [encoder_input, decoder_input],
        decoder_target,
        batch_size=32,
        epochs=100,
        validation_split=0.1,
        verbose=1
    )

    print("\n=== Saving model and tokenizers ===")
    model.save('model.keras')

    with open('tokenizer_ap.pkl', 'wb') as f:
        pickle.dump(tokenizer_ap, f)
    with open('tokenizer_en.pkl', 'wb') as f:
        pickle.dump(tokenizer_en, f)
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump({'max_ap_len': max_ap_len, 'max_en_len': max_en_len,
                     'vocab_size_ap': vocab_size_ap, 'vocab_size_en': vocab_size_en}, f)

    print("Done! Files saved: model.keras, tokenizer_ap.pkl, tokenizer_en.pkl, model_metadata.pkl")


if __name__ == '__main__':
    main()
