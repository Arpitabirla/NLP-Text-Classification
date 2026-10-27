"""
=============================================================
NLP Mini Project — bow.py
=============================================================
Covers:
    1. Bag of Words (BOW) using sklearn CountVectorizer
    2. Manual BOW from scratch (to understand internals)
    3. Vocabulary inspection
    4. Document-Term Matrix visualization
=============================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# ─────────────────────────────────────────────────────────────────────────────
# 1. BOW FROM SCRATCH (Educational)
# ─────────────────────────────────────────────────────────────────────────────

def build_vocabulary(documents: list[str]) -> dict[str, int]:
    """
    Build a vocabulary dictionary from a list of documents.
    Each unique word is assigned a unique integer index.

    Args:
        documents: List of text strings.
    Returns:
        vocabulary dict mapping word -> index.
    """
    vocab = {}
    idx   = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def document_to_bow_vector(document: str, vocabulary: dict[str, int]) -> list[int]:
    """
    Convert a single document to a BOW vector using the given vocabulary.

    Args:
        document: Text string.
        vocabulary: Vocabulary dict from build_vocabulary().
    Returns:
        Integer list of length len(vocabulary) with word counts.
    """
    vector = [0] * len(vocabulary)
    words  = document.lower().split()
    for word in words:
        if word in vocabulary:
            vector[vocabulary[word]] += 1
    return vector


def build_bow_matrix_scratch(documents: list[str]) -> tuple[np.ndarray, dict]:
    """
    Build a full BOW document-term matrix from scratch.

    Args:
        documents: List of text strings.
    Returns:
        Tuple of (matrix as np.ndarray, vocabulary dict).
    """
    vocab  = build_vocabulary(documents)
    matrix = np.array([document_to_bow_vector(doc, vocab) for doc in documents])
    return matrix, vocab


# ─────────────────────────────────────────────────────────────────────────────
# 2. BOW USING SKLEARN (Production-ready)
# ─────────────────────────────────────────────────────────────────────────────

def build_bow_sklearn(documents: list[str], max_features: int = 500,
                      ngram_range: tuple = (1, 1)) -> tuple:
    """
    Build a BOW matrix using sklearn's CountVectorizer.
    Supports unigrams, bigrams, etc.

    Args:
        documents: List of text strings.
        max_features: Maximum number of features (vocabulary size).
        ngram_range: Range of n-grams (1,1)=unigram, (1,2)=unigram+bigram.
    Returns:
        Tuple of (vectorizer, sparse_matrix, feature_names).
    """
    vectorizer = CountVectorizer(
        max_features  = max_features,
        ngram_range   = ngram_range,
        stop_words    = "english",
        strip_accents = "unicode",
        lowercase     = True,
    )
    bow_matrix    = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    return vectorizer, bow_matrix, feature_names


def get_bow_dataframe(documents: list[str], doc_labels: list[str] = None,
                      max_features: int = 30) -> pd.DataFrame:
    """
    Return BOW matrix as a readable Pandas DataFrame.

    Args:
        documents: List of text strings.
        doc_labels: Optional list of document labels for index.
        max_features: Number of top features to include.
    Returns:
        DataFrame with documents as rows and words as columns.
    """
    vectorizer, matrix, features = build_bow_sklearn(documents, max_features)
    df = pd.DataFrame(
        matrix.toarray(),
        columns = features,
        index   = doc_labels if doc_labels else [f"Doc_{i+1}" for i in range(len(documents))]
    )
    return df


def get_top_words_bow(documents: list[str], top_n: int = 20) -> list[tuple]:
    """
    Get the top-N most frequent words across all documents.

    Args:
        documents: List of text strings.
        top_n: Number of top words to return.
    Returns:
        List of (word, count) tuples sorted by frequency.
    """
    all_words = []
    for doc in documents:
        all_words.extend(doc.lower().split())
    counter = Counter(all_words)
    return counter.most_common(top_n)


def word_frequency_by_doc(documents: list[str], doc_labels: list[str] = None) -> pd.DataFrame:
    """
    Compute word frequency stats per document.

    Args:
        documents: List of text strings.
        doc_labels: Optional list of document labels.
    Returns:
        DataFrame with word frequency statistics.
    """
    stats = []
    for i, doc in enumerate(documents):
        words    = doc.lower().split()
        unique   = len(set(words))
        total    = len(words)
        label    = doc_labels[i] if doc_labels else f"Doc_{i+1}"
        stats.append({
            "Document"     : label,
            "Total Words"  : total,
            "Unique Words" : unique,
            "Vocab Density": round(unique / total, 3) if total > 0 else 0,
        })
    return pd.DataFrame(stats)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sample corpus (5 documents)
    docs = [
        "The cat sat on the mat and the cat played with the mat",
        "The dog played fetch in the park on a sunny day",
        "Artificial intelligence is transforming the technology sector rapidly",
        "Machine learning algorithms process large amounts of data efficiently",
        "Natural language processing helps computers understand human text",
    ]

    labels = ["Cat & Mat", "Dog Park", "AI Tech", "ML Data", "NLP Text"]

    print("=" * 65)
    print("  BAG OF WORDS (BOW) DEMO")
    print("=" * 65)

    # -- Manual BOW --
    print("\n[*] Manual BOW (from scratch):")
    mat, vocab = build_bow_matrix_scratch(docs[:3])
    sample_vocab = dict(list(vocab.items())[:10])
    print(f"   Vocabulary (first 10): {sample_vocab}")
    print(f"   BOW Matrix shape: {mat.shape}")
    print(f"   BOW Matrix:\n{mat[:, :10]}")

    # -- Sklearn BOW --
    print("\n[*] sklearn CountVectorizer BOW:")
    vec, bow, features = build_bow_sklearn(docs, max_features=20)
    print(f"   Matrix shape: {bow.shape}")
    print(f"   Features: {features[:15].tolist()}")

    # -- BOW DataFrame --
    df = get_bow_dataframe(docs, labels, max_features=15)
    print(f"\n[*] BOW DataFrame (top 15 words):")
    print(df.to_string())

    # -- Top Words --
    top_words = get_top_words_bow(docs, top_n=10)
    print(f"\n[*] Top-10 Most Frequent Words:")
    print(f"   {'Word':<20} {'Count':<10}")
    print("   " + "-" * 30)
    for word, count in top_words:
        print(f"   {word:<20} {count:<10}")

    # -- Word Frequency Stats --
    stats_df = word_frequency_by_doc(docs, labels)
    print(f"\n[*] Word Frequency Statistics per Document:")
    print(stats_df.to_string(index=False))

