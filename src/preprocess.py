"""
Task 3 — Information Processing.

Takes the raw documents from data/raw/ and produces a clean, de-duplicated
corpus at data/corpus.csv with one row per document:

    id, title, text, source, source_type, url, date

Steps (all classical NLP, Module 2/3 style):
  1. merge all raw sources
  2. clean text (handled in collector; re-applied defensively)
  3. drop empty / very short documents
  4. keep only documents relevant to the company or its competitors
  5. remove duplicates (exact id + identical normalised title)
"""

import re

import pandas as pd

from src import config
from src.utils import clean_text, load_json

MAX_DOCS_PER_TYPE = 120  # balance + shrink: at most this many docs per source type

# Match brand/competitor names as WHOLE WORDS so a generic token ("Tesla", which is
# also a physics unit / a surname) doesn't catch unrelated text. Falls back to COMPANY
# if no alias list is set.
_ALIASES = getattr(config, "COMPANY_ALIASES", [config.COMPANY]) + config.COMPETITORS
_RELEVANT_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ALIASES) + r")\b", re.IGNORECASE)


def _relevant(text: str) -> bool:
    """Keep a document only if it mentions the company (by any alias) or a competitor."""
    return bool(_RELEVANT_RE.search(text))


def build_corpus() -> pd.DataFrame:
    raw = load_json(config.RAW_DIR / "all_raw.json")["documents"]
    df = pd.DataFrame(raw)

    # 2. defensive re-clean
    df["title"] = df["title"].fillna("").map(clean_text)
    df["text"] = df["text"].fillna("").map(clean_text)

    # 3. drop empties / too-short (less than 5 words of text)
    df = df[df["text"].str.split().map(len) >= 5]

    # 4. relevance filter ("extract relevant information")
    df = df[(df["title"] + " " + df["text"]).map(_relevant)]

    # 5. de-duplicate
    df["_norm_title"] = df["title"].str.lower().str.strip()
    df = df.drop_duplicates(subset="id")
    df = df.drop_duplicates(subset="_norm_title")
    df = df.drop(columns="_norm_title").reset_index(drop=True)

    # 6. cap per source type -> smaller, balanced corpus (news/community no longer dominate)
    df = df.groupby("source_type", group_keys=False).head(MAX_DOCS_PER_TYPE).reset_index(drop=True)

    df.to_csv(config.CORPUS_CSV, index=False)
    print(f"[preprocess] clean corpus: {len(df)} documents -> {config.CORPUS_CSV.name}")
    print(f"[preprocess] sources: {df['source_type'].value_counts().to_dict()}")
    return df


def load_corpus() -> pd.DataFrame:
    """Convenience loader used by the knowledge base / agents."""
    return pd.read_csv(config.CORPUS_CSV).fillna("")


if __name__ == "__main__":
    build_corpus()
