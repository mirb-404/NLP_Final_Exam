"""
Task 3 — Information Processing (classical NLP, Module 2/3).

Raw docs (data/raw/) -> clean, de-duplicated corpus (data/corpus.csv). Steps: merge
sources, defensive re-clean, drop <5-word docs, keep only company/competitor-relevant
docs, drop duplicates (id + normalised title), cap per source type for balance.
"""

import itertools
import re

import pandas as pd

from src import config
from src.utils import clean_text, load_json

MAX_DOCS_PER_TYPE = 500  # growth ceiling per source type — accumulation fills toward this
                         # (round-robin first keeps the sources balanced under the cap)

# Match brand/competitor names as WHOLE WORDS so a generic token ("Tesla", which is
# also a physics unit / a surname) doesn't catch unrelated text. Falls back to COMPANY
# if no alias list is set.
_ALIASES = getattr(config, "COMPANY_ALIASES", [config.COMPANY]) + config.COMPETITORS
_RELEVANT_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ALIASES) + r")\b", re.IGNORECASE)


def _relevant(text: str) -> bool:
    """Keep a document only if it mentions the company (by any alias) or a competitor."""
    return bool(_RELEVANT_RE.search(text))


def _round_robin_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder rows so that within each source_type the underlying sources alternate
    (round-robin). The per-type head() cap then keeps a balanced mix instead of dropping
    whichever source was collected last — e.g. Stack Overflow sat entirely behind Hacker
    News in 'community', so the per-type cap dropped all of it."""
    order = []
    for _, grp in df.groupby("source_type", sort=False):
        per_source = [list(g.index) for _, g in grp.groupby("source", sort=False)]
        for picks in itertools.zip_longest(*per_source):
            order.extend(i for i in picks if i is not None)
    return df.loc[order]


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

    # 6. balance the sources within each type, then cap per type -> smaller, balanced corpus
    #    (round-robin first so e.g. 'community' keeps both Hacker News AND Stack Overflow)
    df = _round_robin_by_source(df)
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
