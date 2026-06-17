"""
Classical NLP Agent  (Module 2, 3, 9 idioms).

Provides the "classical" signals the dashboard and intelligence engine need,
without any LLM:

  - sentiment   : VADER (nltk) compound score  -> news vs public sentiment
  - entities    : spaCy NER (organisations, products, people)
  - keywords    : scikit-learn TF-IDF top terms

These are deterministic and fast, so they run over the whole corpus.
"""

from collections import Counter

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config

# ---- one-time lightweight resource downloads -------------------------------
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: E402

_VADER = SentimentIntensityAnalyzer()

# spaCy is optional: if the model is missing we degrade gracefully.
try:
    import spacy

    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None
    print("[classical_agent] spaCy model 'en_core_web_sm' not found — "
          "run: uv run python -m spacy download en_core_web_sm")


# ----------------------------------------------------------------------------
# Sentiment (Module 9)
# ----------------------------------------------------------------------------
def analyze_sentiment(text: str) -> dict:
    """VADER compound score -> {label, score}. score in [-1, 1]."""
    score = _VADER.polarity_scores(text or "")["compound"]
    label = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
    return {"label": label, "score": round(score, 4)}


def corpus_sentiment(df) -> dict:
    """
    Aggregate sentiment for the dashboard (Section 5).
    'news'/'finance' -> news sentiment ; 'community' -> public sentiment.
    """
    scores = df["title"].fillna("").map(lambda t: _VADER.polarity_scores(t)["compound"])
    df = df.assign(_sent=scores)

    def _avg(mask):
        sub = df[mask]
        return round(float(sub["_sent"].mean()), 4) if len(sub) else 0.0

    news_mask = df["source_type"].isin(["news", "finance"])
    public_mask = df["source_type"] == "community"
    return {
        "news_sentiment": _avg(news_mask),
        "public_sentiment": _avg(public_mask),
        "overall_sentiment": round(float(df["_sent"].mean()), 4),
        "distribution": Counter(
            df["_sent"].map(
                lambda s: "positive" if s > 0.05 else "negative" if s < -0.05 else "neutral"
            )
        ),
    }


# ----------------------------------------------------------------------------
# Named entities (Module 3)
# ----------------------------------------------------------------------------
def extract_entities(text: str) -> dict:
    """Return organisations / products / people mentioned (spaCy NER)."""
    if _NLP is None:
        return {"ORG": [], "PRODUCT": [], "PERSON": []}
    doc = _NLP(text[:2000])  # cap length for speed
    out = {"ORG": [], "PRODUCT": [], "PERSON": []}
    for ent in doc.ents:
        if ent.label_ in out:
            out[ent.label_].append(ent.text)
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}  # dedupe, keep order


# ----------------------------------------------------------------------------
# TF-IDF keywords (Module 4/10)
# ----------------------------------------------------------------------------
def top_keywords(texts: list[str], n: int = 15) -> list[str]:
    """Most important terms across a set of documents (TF-IDF)."""
    if not texts:
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    matrix = vec.fit_transform(texts)
    weights = matrix.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    ranked = sorted(zip(vocab, weights), key=lambda x: x[1], reverse=True)
    return [term for term, _ in ranked[:n]]


if __name__ == "__main__":
    from src.preprocess import load_corpus

    df = load_corpus()
    print("Sentiment:", corpus_sentiment(df))
    print("Keywords:", top_keywords((df["title"] + " " + df["text"]).tolist()))
