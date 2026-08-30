import re
import collections
import json

from config import ALPHABET, DATA_PATH


def initialize_vocabulary(text):
    vocab = collections.defaultdict(int)
    words = text.strip().split()
    for word in words:
        vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab


def get_tokens_and_frequencies(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens


def get_pairs_and_counts(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs


def merge_pair_in_vocabulary(pair, vocab_in):
    vocab_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab_in:
        word_out = p.sub(''.join(pair), word)
        vocab_out[word_out] = vocab_in[word]
    return vocab_out


def tokenize(text, num_merges):
    # Initialize the vocabulary from the input text
    vocab = initialize_vocabulary(text)

    for i in range(num_merges):
        # Find the tokens and how often they occur in the vocabulary
        tokens = get_tokens_and_frequencies(vocab)

        # Find the pairs of adjacent tokens and their counts
        pairs = get_pairs_and_counts(vocab)

        # Find the most frequent pair
        most_frequent_pair = max(pairs, key=pairs.get)
        print('Most frequent pair: {}'.format(most_frequent_pair))

        # Merge the code in the vocabulary
        vocab = merge_pair_in_vocabulary(most_frequent_pair, vocab)

    # Find the tokens and how often they occur in the vocabulary one last time
    tokens = get_tokens_and_frequencies(vocab)

    return tokens, vocab


with open(DATA_PATH, "r", encoding="utf-8") as f:
    songs = json.load(f)

text = ""
for song in songs:
    text += song.get("lines", [])

tokens, vocab = tokenize(text, num_merges=50)
print('Tokens: {}'.format(tokens))
print('Number of tokens: {}'.format(len(tokens)))
print('Vocabulary: {}'.format(vocab))
print('Size of vocabulary: {}'.format(len(vocab)))
