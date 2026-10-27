"""
=============================================================
NLP Mini Project — tfidf.py
=============================================================
Covers:
    1. TF-IDF from scratch (manual formula)
    2. TF-IDF using sklearn TfidfVectorizer
    3. Top important terms per document
    4. TF-IDF vs BOW comparison
=============================================================

TF-IDF Formula:
    TF  (term frequency)  = count(word in doc) / total words in doc
    IDF (inverse doc freq) = log(N / (1 + df)) + 1
                               where N = total docs, df = docs containing word
    TF-IDF = TF × IDF
=============================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────────────────────────────────────
# 1. TF-IDF FROM SCRATCH (Educational)
# ─────────────────────────────────────────────────────────────────────────────

def compute_tf(document: str) -> dict[str, float]:
    """
    Compute Term Frequency (TF) for a single document.
    TF = count(word) / total_words_in_document

    Args:
        document: Text string.
    Returns:
        Dict mapping word -> TF score.
    """
    words = document.lower().split()
    total = len(words)
    freq  = Counter(words)
    return {word: count / total for word, count in freq.items()}


def compute_idf(documents: list[str]) -> dict[str, float]:
    """
    Compute Inverse Document Frequency (IDF) for a corpus.
    IDF = log(N / (1 + df)) + 1  (sklearn-style smoothed IDF)

    Args:
        documents: List of text strings.
    Returns:
        Dict mapping word -> IDF score.
    """
    N  = len(documents)
    df = {}  # document frequency for each word

    for doc in documents:
        words = set(doc.lower().split())
        for word in words:
            df[word] = df.get(word, 0) + 1

    idf = {word: math.log(N / (1 + count)) + 1 for word, count in df.items()}
    return idf


def compute_tfidf(document: str, idf: dict[str, float]) -> dict[str, float]:
    """
    Compute TF-IDF scores for a single document.

    Args:
        document: Text string.
        idf: IDF dictionary from compute_idf().
    Returns:
        Dict mapping word -> TF-IDF score (sorted descending).
    """
    tf = compute_tf(document)
    tfidf = {word: tf_score * idf.get(word, 0)
             for word, tf_score in tf.items()}
    return dict(sorted(tfidf.items(), key=lambda x: x[1], reverse=True))


def build_tfidf_matrix_scratch(documents: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Build a full TF-IDF matrix from scratch for a list of documents.

    Args:
        documents: List of text strings.
    Returns:
        Tuple of (tfidf_matrix as np.ndarray, vocabulary list).
    """
    # Build global vocabulary
    vocab = sorted(set(
        word for doc in documents for word in doc.lower().split()
    ))
    idf = compute_idf(documents)

    matrix = []
    for doc in documents:
        tf     = compute_tf(doc)
        vector = [tf.get(word, 0) * idf.get(word, 0) for word in vocab]
        matrix.append(vector)

    return np.array(matrix), vocab


# ─────────────────────────────────────────────────────────────────────────────
# 2. TF-IDF USING SKLEARN (Production-ready)
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_sklearn(documents: list[str], max_features: int = 500,
                        ngram_range: tuple = (1, 1)) -> tuple:
    """
    Build a TF-IDF matrix using sklearn's TfidfVectorizer.

    Args:
        documents: List of text strings.
        max_features: Maximum number of features.
        ngram_range: N-gram range for feature extraction.
    Returns:
        Tuple of (vectorizer, sparse_matrix, feature_names).
    """
    vectorizer = TfidfVectorizer(
        max_features  = max_features,
        ngram_range   = ngram_range,
        stop_words    = "english",
        strip_accents = "unicode",
        lowercase     = True,
        sublinear_tf  = True,   # Apply sublinear TF scaling: 1 + log(TF)
    )
    tfidf_matrix  = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    return vectorizer, tfidf_matrix, feature_names


def get_tfidf_dataframe(documents: list[str], doc_labels: list[str] = None,
                        max_features: int = 30) -> pd.DataFrame:
    """
    Return TF-IDF matrix as a readable Pandas DataFrame.

    Args:
        documents: List of text strings.
        doc_labels: Optional list of document labels.
        max_features: Number of top features.
    Returns:
        DataFrame with documents as rows, terms as columns.
    """
    vectorizer, matrix, features = build_tfidf_sklearn(documents, max_features)
    df = pd.DataFrame(
        matrix.toarray().round(4),
        columns = features,
        index   = doc_labels if doc_labels else [f"Doc_{i+1}" for i in range(len(documents))]
    )
    return df


def get_top_terms_per_doc(documents: list[str], doc_labels: list[str] = None,
                          top_n: int = 5) -> pd.DataFrame:
    """
    Get the top-N most important TF-IDF terms for each document.

    Args:
        documents: List of text strings.
        doc_labels: Optional list of document labels.
        top_n: Number of top terms to show per document.
    Returns:
        DataFrame with top terms per document.
    """
    idf = compute_idf(documents)
    rows = []
    for i, doc in enumerate(documents):
        tfidf  = compute_tfidf(doc, idf)
        top    = list(tfidf.items())[:top_n]
        label  = doc_labels[i] if doc_labels else f"Doc_{i+1}"
        for rank, (word, score) in enumerate(top, 1):
            rows.append({
                "Document" : label,
                "Rank"     : rank,
                "Term"     : word,
                "TF-IDF"   : round(score, 4),
            })
    return pd.DataFrame(rows)


def compare_bow_vs_tfidf(documents: list[str], top_n: int = 10) -> pd.DataFrame:
    """
    Compare BOW and TF-IDF scores for the same corpus.

    Args:
        documents: List of text strings.
        top_n: Number of top features to compare.
    Returns:
        DataFrame showing BOW count vs TF-IDF score for top terms.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    bow_vec   = CountVectorizer(max_features=top_n, stop_words="english")
    tfidf_vec = TfidfVectorizer(max_features=top_n, stop_words="english")

    bow_mat   = bow_vec.fit_transform(documents).toarray()
    tfidf_mat = tfidf_vec.fit_transform(documents).toarray()

    bow_totals   = bow_mat.sum(axis=0)
    tfidf_totals = tfidf_mat.sum(axis=0)

    bow_df   = pd.DataFrame({"Term": bow_vec.get_feature_names_out(),   "BOW Count": bow_totals})
    tfidf_df = pd.DataFrame({"Term": tfidf_vec.get_feature_names_out(), "TF-IDF Sum": tfidf_totals.round(4)})

    merged = pd.merge(bow_df, tfidf_df, on="Term", how="outer").fillna(0)
    return merged.sort_values("BOW Count", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs = [
        "The cat sat on the mat and the cat played with the mat",
        "The dog played fetch in the park on a sunny day",
        "Artificial intelligence is transforming the technology sector rapidly",
        "Machine learning algorithms process large amounts of data efficiently",
        "Natural language processing helps computers understand human text",
    ]
    labels = ["Cat & Mat", "Dog Park", "AI Tech", "ML Data", "NLP Text"]

    print("=" * 65)
    print("  TF-IDF DEMO")
    print("=" * 65)

    # -- TF for first doc --
    tf = compute_tf(docs[0])
    print(f"\n[*] TF scores for '{labels[0]}':")
    for word, score in sorted(tf.items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"   {word:<20} {score:.4f}")

    # -- IDF --
    idf = compute_idf(docs)
    print(f"\n[*] IDF scores (sample 8 words):")
    for word, score in list(sorted(idf.items(), key=lambda x: x[1], reverse=True))[:8]:
        print(f"   {word:<20} {score:.4f}")

    # -- TF-IDF for first doc --
    tfidf = compute_tfidf(docs[0], idf)
    print(f"\n[*] TF-IDF scores for '{labels[0]}':")
    for word, score in list(tfidf.items())[:8]:
        print(f"   {word:<20} {score:.4f}")

    # -- sklearn TF-IDF --
    print(f"\n[*] sklearn TF-IDF Matrix:")
    df = get_tfidf_dataframe(docs, labels, max_features=15)
    print(df.to_string())

    # -- Top Terms per Doc --
    top_df = get_top_terms_per_doc(docs, labels, top_n=3)
    print(f"\n[*] Top-3 TF-IDF Terms per Document:")
    print(top_df.to_string(index=False))

    # -- BOW vs TF-IDF --
    cmp_df = compare_bow_vs_tfidf(docs, top_n=10)
    print(f"\n[*] BOW vs TF-IDF Comparison:")
    print(cmp_df.to_string(index=False))

