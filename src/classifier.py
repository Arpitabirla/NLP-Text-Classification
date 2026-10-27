"""
=============================================================
NLP Mini Project — classifier.py
=============================================================
Covers:
    1. Text classification using BOW features (Naive Bayes)
    2. Text classification using TF-IDF features (Logistic Regression)
    3. Model evaluation: accuracy, classification report, confusion matrix
    4. Comparison of BOW vs TF-IDF for classification
=============================================================
Dataset: NLTK Movie Reviews (Positive vs Negative)
=============================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

# ── Download Movie Reviews dataset ───────────────────────────────────────────
def download_resources():
    nltk.download("movie_reviews", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

download_resources()

from nltk.corpus import movie_reviews


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_movie_reviews() -> tuple[list[str], list[str]]:
    """
    Load the NLTK Movie Reviews dataset.
    Contains 1000 positive and 1000 negative reviews.

    Returns:
        Tuple of (documents list, labels list) where labels are 'pos'/'neg'.
    """
    documents = []
    labels    = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            # Join all words in the review into a single string
            text = " ".join(movie_reviews.words(fileid))
            documents.append(text)
            labels.append(category)
    return documents, labels


def get_train_test_split(documents: list[str], labels: list[str],
                         test_size: float = 0.2, random_state: int = 42):
    """
    Split corpus into training and test sets.

    Args:
        documents: List of text strings.
        labels: List of category labels.
        test_size: Fraction for test set.
        random_state: Random seed for reproducibility.
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        documents, labels,
        test_size    = test_size,
        random_state = random_state,
        stratify     = labels,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIPELINES: BOW + Naive Bayes
# ─────────────────────────────────────────────────────────────────────────────

def build_bow_nb_pipeline(max_features: int = 5000, ngram_range: tuple = (1, 2)) -> Pipeline:
    """
    Build a classification pipeline: BOW → Multinomial Naive Bayes.

    Args:
        max_features: Vocabulary size limit.
        ngram_range: N-gram range for CountVectorizer.
    Returns:
        sklearn Pipeline object.
    """
    return Pipeline([
        ("bow", CountVectorizer(
            max_features  = max_features,
            ngram_range   = ngram_range,
            stop_words    = "english",
            strip_accents = "unicode",
            lowercase     = True,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 3. PIPELINES: TF-IDF + Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_lr_pipeline(max_features: int = 5000, ngram_range: tuple = (1, 2)) -> Pipeline:
    """
    Build a classification pipeline: TF-IDF → Logistic Regression.

    Args:
        max_features: Vocabulary size limit.
        ngram_range: N-gram range for TfidfVectorizer.
    Returns:
        sklearn Pipeline object.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features  = max_features,
            ngram_range   = ngram_range,
            stop_words    = "english",
            strip_accents = "unicode",
            lowercase     = True,
            sublinear_tf  = True,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])


def build_tfidf_svm_pipeline(max_features: int = 5000) -> Pipeline:
    """
    Build a classification pipeline: TF-IDF → Linear SVM.

    Args:
        max_features: Vocabulary size limit.
    Returns:
        sklearn Pipeline object.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features = max_features,
            stop_words   = "english",
            sublinear_tf = True,
        )),
        ("clf", LinearSVC(C=1.0, random_state=42)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(pipeline: Pipeline, X_train, X_test, y_train, y_test,
                   model_name: str = "Model") -> dict:
    """
    Train and evaluate a classification pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train, X_test: Text documents.
        y_train, y_test: Labels.
        model_name: Display name for the model.
    Returns:
        Dict with accuracy, report, and confusion matrix.
    """
    pipeline.fit(X_train, y_train)
    y_pred   = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  [OK] Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n  Classification Report:")
    print(report)
    print(f"  Confusion Matrix:")
    print(f"  {cm}")

    return {
        "model_name": model_name,
        "pipeline"  : pipeline,
        "accuracy"  : accuracy,
        "report"    : report,
        "cm"        : cm,
        "y_pred"    : y_pred,
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Create a comparison DataFrame for multiple evaluated models.

    Args:
        results: List of result dicts from evaluate_model().
    Returns:
        DataFrame comparing model accuracies.
    """
    rows = [{"Model": r["model_name"], "Accuracy (%)": round(r["accuracy"] * 100, 2)}
            for r in results]
    df = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False)
    return df.reset_index(drop=True)


def predict_sentiment(text: str, pipeline: Pipeline) -> str:
    """
    Predict sentiment label for a new text string.

    Args:
        text: Input text to classify.
        pipeline: Trained sklearn Pipeline.
    Returns:
        Predicted label string.
    """
    prediction = pipeline.predict([text])[0]
    return prediction


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  TEXT CLASSIFICATION DEMO (NLTK Movie Reviews)")
    print("=" * 65)

    # -- Load Data --
    print("\n[*] Loading NLTK Movie Reviews dataset...")
    docs, labels = load_movie_reviews()
    print(f"   Total reviews: {len(docs)}")
    print(f"   Categories: {set(labels)}")

    X_train, X_test, y_train, y_test = get_train_test_split(docs, labels)
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Model 1: BOW + Naive Bayes ──
    bow_nb = build_bow_nb_pipeline()
    result1 = evaluate_model(bow_nb, X_train, X_test, y_train, y_test,
                             model_name="BOW + Multinomial Naive Bayes")

    # ── Model 2: TF-IDF + Logistic Regression ──
    tfidf_lr = build_tfidf_lr_pipeline()
    result2  = evaluate_model(tfidf_lr, X_train, X_test, y_train, y_test,
                              model_name="TF-IDF + Logistic Regression")

    # ── Model 3: TF-IDF + SVM ──
    tfidf_svm = build_tfidf_svm_pipeline()
    result3   = evaluate_model(tfidf_svm, X_train, X_test, y_train, y_test,
                               model_name="TF-IDF + Linear SVM")

    # ── Comparison ──
    cmp_df = compare_models([result1, result2, result3])
    print(f"\n{'='*55}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(cmp_df.to_string(index=False))

    # ── Live Prediction ──
    print(f"\n{'='*55}")
    print(f"  LIVE PREDICTIONS (TF-IDF + LR)")
    print(f"{'='*55}")
    test_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the story was brilliant.",
        "Terrible film. Complete waste of time. The plot made no sense and acting was awful.",
        "It was okay, not great but not terrible either. Some good moments.",
    ]
    for review in test_reviews:
        label = predict_sentiment(review, tfidf_lr)
        sentiment = "[POSITIVE]" if label == "pos" else "[NEGATIVE]"
        print(f"\n  {sentiment} Predicted: {label.upper()}")
        print(f"     Text: {review[:70]}...")
