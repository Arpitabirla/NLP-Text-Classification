"""
=============================================================
NLP Mini Project — preprocessing.py
=============================================================
Covers:
    1. Tokenization  (Word + Sentence)
    2. Stopword Removal
    3. Stemming      (PorterStemmer + SnowballStemmer)
    4. Lemmatization (WordNetLemmatizer with POS tags)
=============================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import nltk
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# ── Download required NLTK resources (only once) ─────────────────────────────
def download_nltk_resources():
    """Download all required NLTK data packages."""
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for r in resources:
        nltk.download(r, quiet=True)

download_nltk_resources()

# ─────────────────────────────────────────────────────────────────────────────
# 1. TOKENIZATION
# ─────────────────────────────────────────────────────────────────────────────

def word_tokenize_text(text: str) -> list[str]:
    """
    Tokenize text into individual words using NLTK's word_tokenize.
    Handles contractions, punctuation, and special characters.

    Args:
        text: Input text string.
    Returns:
        List of word tokens.
    """
    tokens = word_tokenize(text)
    return tokens


def sentence_tokenize_text(text: str) -> list[str]:
    """
    Tokenize text into sentences using NLTK's sent_tokenize.

    Args:
        text: Input text string.
    Returns:
        List of sentence strings.
    """
    sentences = sent_tokenize(text)
    return sentences


def clean_tokens(tokens: list[str], remove_stopwords: bool = True,
                 remove_punct: bool = True) -> list[str]:
    """
    Clean a list of tokens by removing stopwords and punctuation.

    Args:
        tokens: List of word tokens.
        remove_stopwords: Whether to remove English stopwords.
        remove_punct: Whether to remove punctuation tokens.
    Returns:
        Cleaned list of tokens (lowercased).
    """
    stop_words = set(stopwords.words("english")) if remove_stopwords else set()
    punct = set(string.punctuation)

    cleaned = []
    for token in tokens:
        token_lower = token.lower()
        if remove_punct and token_lower in punct:
            continue
        if remove_stopwords and token_lower in stop_words:
            continue
        if not token.isalpha():  # skip numbers, special chars
            continue
        cleaned.append(token_lower)
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# 2. STEMMING
# ─────────────────────────────────────────────────────────────────────────────

def stem_porter(tokens: list[str]) -> list[str]:
    """
    Apply Porter Stemming to a list of tokens.
    Porter Stemmer is a simple, fast rule-based stemmer.

    Args:
        tokens: List of word tokens.
    Returns:
        List of porter-stemmed tokens.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def stem_snowball(tokens: list[str], language: str = "english") -> list[str]:
    """
    Apply Snowball (Lancaster) Stemming to a list of tokens.
    More aggressive than Porter Stemmer.

    Args:
        tokens: List of word tokens.
        language: Language for Snowball stemmer (default: 'english').
    Returns:
        List of snowball-stemmed tokens.
    """
    stemmer = SnowballStemmer(language)
    return [stemmer.stem(token) for token in tokens]


def compare_stemmers(tokens: list[str]) -> dict:
    """
    Compare Porter and Snowball stemming side-by-side.

    Args:
        tokens: List of word tokens.
    Returns:
        Dict with original, porter_stemmed, and snowball_stemmed lists.
    """
    return {
        "original"       : tokens,
        "porter_stemmed" : stem_porter(tokens),
        "snowball_stemmed": stem_snowball(tokens),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. LEMMATIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _get_wordnet_pos(treebank_tag: str) -> str:
    """
    Convert POS tag from Penn Treebank format to WordNet format.
    This improves lemmatization accuracy significantly.

    Args:
        treebank_tag: POS tag string (e.g., 'VBG', 'NNS').
    Returns:
        WordNet POS constant.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """
    Lemmatize tokens using WordNetLemmatizer with POS-tag awareness.
    POS-aware lemmatization gives much better results than basic lemmatization.

    Args:
        tokens: List of word tokens.
    Returns:
        List of lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()
    pos_tags   = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(word, _get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]
    return lemmatized


def compare_stem_vs_lemma(tokens: list[str]) -> dict:
    """
    Compare stemming vs lemmatization for a list of tokens.

    Args:
        tokens: List of word tokens.
    Returns:
        Dict showing original, stemmed, and lemmatized forms.
    """
    return {
        "original"   : tokens,
        "porter_stem": stem_porter(tokens),
        "snowball_stem": stem_snowball(tokens),
        "lemmatized" : lemmatize_tokens(tokens),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_text = (
        "Natural language processing is an exciting field of artificial intelligence. "
        "Machines are learning to understand and generate human languages. "
        "The running dogs were happily playing in the beautiful gardens."
    )

    print("=" * 65)
    print("  NLP PREPROCESSING DEMO")
    print("=" * 65)

    # -- Sentence Tokenization --
    sentences = sentence_tokenize_text(sample_text)
    print(f"\n[*] Sentence Tokenization ({len(sentences)} sentences):")
    for i, s in enumerate(sentences, 1):
        print(f"   {i}. {s}")

    # -- Word Tokenization --
    raw_tokens = word_tokenize_text(sample_text)
    print(f"\n[*] Word Tokenization ({len(raw_tokens)} tokens):")
    print(f"   {raw_tokens}")

    # -- Clean Tokens --
    clean = clean_tokens(raw_tokens)
    print(f"\n[*] After Cleaning (stopwords + punct removed) -- {len(clean)} tokens:")
    print(f"   {clean}")

    # -- Stemming Comparison --
    comparison = compare_stemmers(clean)
    print("\n[*] Stemming Comparison:")
    print(f"   {'Word':<20} {'Porter':<20} {'Snowball':<20}")
    print("   " + "-" * 60)
    for orig, porter, snow in zip(
        comparison["original"],
        comparison["porter_stemmed"],
        comparison["snowball_stemmed"],
    ):
        print(f"   {orig:<20} {porter:<20} {snow:<20}")

    # -- Lemmatization --
    lemmatized = lemmatize_tokens(clean)
    print(f"\n[*] Lemmatization (POS-aware):")
    print(f"   Original  : {clean}")
    print(f"   Lemmatized: {lemmatized}")

    # -- Side-by-Side Comparison --
    full = compare_stem_vs_lemma(clean)
    print("\n[*] Stemming vs Lemmatization (Side-by-Side):")
    print(f"   {'Word':<20} {'Porter':<15} {'Snowball':<15} {'Lemma':<15}")
    print("   " + "-" * 65)
    for o, p, s, l in zip(
        full["original"], full["porter_stem"],
        full["snowball_stem"], full["lemmatized"]
    ):
        print(f"   {o:<20} {p:<15} {s:<15} {l:<15}")

