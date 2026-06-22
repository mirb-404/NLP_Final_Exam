"""
Classical NLP Agent  (Module 3 / 9 idioms).

Provides the "classical" signals the dashboard and intelligence engine need,
without the reasoning LLM:

  - sentiment   : RoBERTa fine-tuned on the 3-class TweetEval scheme
                  (negative/neutral/positive) via a transformers `sentiment-analysis`
                  pipeline. TweetEval is the sentiment task taught in Module 9b; the
                  neutral class is essential so factual finance/news text isn't forced
                  to a pole. Mapped to a signed P(pos)-P(neg) score in [-1, 1].
  - keywords    : scikit-learn TF-IDF top terms

These are deterministic (greedy argmax, no sampling) and run over the whole corpus.
"""

from collections import Counter
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config


# ----------------------------------------------------------------------------
# Sentiment (Module 9b — transformers pipeline, RoBERTa fine-tuned on 3-class TweetEval)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _classifier():
    """Load the 3-class TweetEval sentiment pipeline once (cached)."""
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=config.SENTIMENT_MODEL)


def _signed(scores: list[dict]) -> float:
    """Map one pipeline result `[{label, score}, ...]` to a compound-style value in
    [-1, 1] as `P(positive) − P(negative)`.

    A confident positive -> ~+1, a confident negative -> ~-1, and a neutral doc (most
    mass on the neutral class) -> ~0. This works for the 3-class TweetEval scheme
    (negative/neutral/positive) and degrades correctly to binary models too. Label names
    are matched loosely so both readable labels and `LABEL_0/1/2` schemes work."""
    p = {s["label"].lower(): s["score"] for s in scores}

    def prob(*keys: str) -> float:
        return next((p[k] for k in keys if k in p), 0.0)

    return prob("positive", "pos", "label_2") - prob("negative", "neg", "label_0")


def _sentiment_scores(texts: list[str]) -> list[float]:
    """Signed sentiment in [-1, 1] for each text. One batched forward pass;
    `top_k=None` returns both class probabilities, `truncation` respects the
    model's 512-token limit."""
    if not texts:
        return []
    out = _classifier()([t or " " for t in texts], top_k=None, truncation=True, batch_size=16)
    return [round(_signed(item), 4) for item in out]


def corpus_sentiment(df) -> dict:
    """
    Aggregate sentiment for the dashboard (Section 5).
    'news'/'finance' -> news sentiment ; 'community' -> public sentiment.
    Scores title + body so the signal reflects the whole document, not just the headline.
    """
    blob = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.slice(0, 1000)
    df = df.assign(_sent=_sentiment_scores(blob.tolist()))

    def _avg(mask):
        sub = df[mask]
        return round(float(sub["_sent"].mean()), 4) if len(sub) else 0.0

    news_mask = df["source_type"].isin(["news", "finance"])
    public_mask = df["source_type"] == "community"
    return {
        "news_sentiment": _avg(news_mask),
        "public_sentiment": _avg(public_mask),
        "overall_sentiment": round(float(df["_sent"].mean()), 4) if len(df) else 0.0,
        "distribution": dict(Counter(
            df["_sent"].map(
                lambda s: "positive" if s > 0.05 else "negative" if s < -0.05 else "neutral"
            )
        )),
        "trend": _sentiment_trend(df),
    }


def _sentiment_trend(df) -> list[dict]:
    """Average sentiment per month (Section 5 trend chart). Best-effort on messy feed dates."""
    month = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.to_period("M")
    sub = df.assign(_month=month).dropna(subset=["_month"])
    if sub.empty:
        return []
    g = sub.groupby("_month")["_sent"].agg(["mean", "size"]).reset_index()
    return [{"period": str(r["_month"]), "sentiment": round(float(r["mean"]), 4), "count": int(r["size"])}
            for _, r in g.sort_values("_month").iterrows()]


# ----------------------------------------------------------------------------
# TF-IDF keywords (Module 4/10)
# ----------------------------------------------------------------------------
def top_keywords(texts: list[str], n: int = 15, exclude_domain: bool = True) -> list[str]:
    """Most important terms across a set of documents (TF-IDF).

    With exclude_domain (default), domain noise is dropped — the company's own
    name/ticker/products, news-source names, and generic finance filler — so the
    result is the actual emerging topics, not 'tesla' / 'tsla' / 'yahoo finance'.
    """
    if not texts:
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    matrix = vec.fit_transform(texts)
    weights = matrix.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    ranked = sorted(zip(vocab, weights), key=lambda x: x[1], reverse=True)
    out = []
    for term, _ in ranked:
        if exclude_domain and config.keyword_is_noise(term):
            continue
        out.append(term)
        if len(out) >= n:
            break
    return out


if __name__ == "__main__":
    from src.preprocess import load_corpus

    df = load_corpus()
    print("Sentiment:", corpus_sentiment(df))
    print("Keywords:", top_keywords((df["title"] + " " + df["text"]).tolist()))
