# 🧠 NLP Mini Project

A comprehensive Natural Language Processing (NLP) mini project demonstrating all core text preprocessing and feature extraction techniques.

## 📋 Techniques Covered

| Technique | Method | Library |
|-----------|--------|---------|
| **Tokenization** | Word + Sentence tokenization | `nltk` |
| **Stopword Removal** | English stopwords | `nltk.corpus.stopwords` |
| **Stemming** | Porter Stemmer + Snowball Stemmer | `nltk.stem` |
| **Lemmatization** | POS-aware WordNet Lemmatization | `nltk.stem.WordNetLemmatizer` |
| **Bag of Words** | From scratch + sklearn CountVectorizer | `sklearn` |
| **TF-IDF** | From scratch + sklearn TfidfVectorizer | `sklearn` |
| **Classification** | Naive Bayes, Logistic Regression, SVM | `sklearn` |

---

## 📁 Project Structure

```
NLP/
├── data/
│   └── corpus.txt              # Multi-topic sample corpus
├── src/
│   ├── preprocessing.py        # Tokenization, Stemming, Lemmatization
│   ├── bow.py                  # Bag of Words (from scratch + sklearn)
│   ├── tfidf.py                # TF-IDF (from scratch + sklearn)
│   └── classifier.py          # Text Classification using BOW & TF-IDF
├── app/
│   ├── index.html              # Interactive web dashboard
│   ├── style.css               # Dark-mode premium UI
│   └── app.js                  # NLP algorithms in JavaScript
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Individual Modules
```bash
# Tokenization, Stemming, Lemmatization
python src/preprocessing.py

# Bag of Words
python src/bow.py

# TF-IDF
python src/tfidf.py

# Text Classification (downloads NLTK Movie Reviews ~3MB)
python src/classifier.py
```

### 3. Open Web Dashboard
Simply open `app/index.html` in your browser — no server required!

---

## 📖 Concept Explanations

### 🔤 Tokenization
Breaking text into individual units (words/sentences).
```
Input:  "The cat sat on the mat."
Words:  ["The", "cat", "sat", "on", "the", "mat", "."]
Sentences: ["The cat sat on the mat."]
```

### ✂️ Stemming
Reducing words to their root form using rules.
```
"running" → "run"
"beautiful" → "beauti"  (Porter — may not be a real word)
```

### 📖 Lemmatization
Context-aware reduction to dictionary base form.
```
"running" → "run"   (verb context)
"better"  → "good"  (adjective context)
```

### 🧮 Bag of Words (BOW)
Represent a document as a vector of word counts.
```
Vocab:    [cat, dog, mat, sat]
"cat sat on the mat" → [1, 0, 1, 1]
```

### ⚖️ TF-IDF
Weight words by how important they are to a document.
```
TF  = count(word in doc) / total_words_in_doc
IDF = log(N / (1 + df(word))) + 1
TF-IDF = TF × IDF
```

---

## 🏆 Classification Results

The classifier is trained on the NLTK Movie Reviews dataset (2000 reviews, pos/neg).

| Model | Feature | Expected Accuracy |
|-------|---------|------------------|
| Multinomial Naive Bayes | BOW | ~83% |
| Logistic Regression | TF-IDF | ~87% |
| Linear SVM | TF-IDF | ~88% |

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **NLTK** — Tokenization, Stemming, Lemmatization, Corpus
- **scikit-learn** — CountVectorizer, TfidfVectorizer, Classifiers
- **NumPy & Pandas** — Numerical operations
- **Matplotlib & Seaborn** — Visualizations
- **HTML/CSS/JS** — Interactive dashboard

---

*NLP Mini Project — Demonstrating Core NLP Techniques*
