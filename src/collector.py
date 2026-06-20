"""
Task 1 — Live Data Collection.

Pulls live public documents about the company from four independent source types
and writes them to data/raw/*.json. Every record has the same shape so the rest
of the pipeline is uniform:  {id, title, text, source, source_type, url, date}

    news       Google News RSS
    finance    Yahoo Finance RSS
    community  Hacker News + Stack Overflow
    research   arXiv + OpenAlex
"""

import hashlib

import feedparser
import requests

from src import config
from src.utils import clean_text, now_iso, save_json

# --- source endpoints + queries ---------------------------------------------
HEADERS = {"User-Agent": "ai-ceo-research-agent/1.0 (educational NLP project)"}
TIMEOUT = 25
MIN_DOCS = 100  # PDF Task 1 minimum

# Queries say "Apple Inc" / product names, never bare "Apple", to keep the fruit out.
NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
NEWS_QUERIES = ["Apple Inc", "Apple iPhone", "Apple Vision Pro",
                "Apple AI", "Tim Cook Apple"]

FINANCE_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
# Yahoo's headline RSS alone is thin, so we also pull finance-focused news (tagged finance).
FINANCE_NEWS_QUERIES = ["AAPL stock", "Apple earnings", "Apple stock forecast",
                        "Apple revenue", "Apple dividend"]

HN_API = "https://hn.algolia.com/api/v1/search?query={query}&tags=story&hitsPerPage=60"
HN_QUERIES = ["Apple Inc", "Apple iPhone", "Apple Silicon"]

STACKEX_API = ("https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance"
               "&q={query}&site=stackoverflow&pagesize=40&filter=withbody")
STACKEX_QUERIES = ["Apple iOS", "Apple Swift", "Apple Xcode"]

ARXIV_API = "http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=30"
ARXIV_QUERIES = ["Apple Inc machine learning", "Apple Silicon chip", "Apple iOS privacy"]

OPENALEX_API = "https://api.openalex.org/works?search={query}&per_page=30&mailto=bhandarimirang03@gmail.com"
OPENALEX_QUERIES = ["Apple Inc smartphone", "Apple Silicon processor", "Apple iOS security"]


# --- shared helpers ---------------------------------------------------------
def _quote(text: str) -> str:
    return requests.utils.quote(text)


def _record(title, text, source, source_type, url, date) -> dict:
    """Normalise any source into the uniform document shape."""
    title = clean_text(title)
    return {
        "id": hashlib.md5(f"{url or title}|{title}".encode("utf-8")).hexdigest()[:12],
        "title": title,
        "text": clean_text(text) or title,
        "source": source,
        "source_type": source_type,
        "url": url or "",
        "date": date or "",
    }


def _report(label: str, docs: list[dict]) -> list[dict]:
    print(f"[collector] {label}: {len(docs)} items")
    return docs


def _get_json(url: str) -> dict:
    """GET JSON, returning {} on any failure (so one dead source never aborts the run)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[collector] request failed: {exc}")
        return {}


# --- news ----------------------------------------------------------------
def collect_news() -> list[dict]:
    docs = []
    for query in NEWS_QUERIES:
        for e in feedparser.parse(NEWS_RSS.format(query=_quote(query))).entries:
            docs.append(_record(e.get("title", ""), e.get("summary", ""),
                                e.get("source", {}).get("title", "Google News"),
                                "news", e.get("link", ""), e.get("published", "")))
    return _report("news", docs)


# --- finance -------------------------------------------------------------
def collect_finance() -> list[dict]:
    feed = feedparser.parse(FINANCE_RSS.format(ticker=config.TICKER))
    docs = [_record(e.get("title", ""), e.get("summary", ""), "Yahoo Finance",
                    "finance", e.get("link", ""), e.get("published", ""))
            for e in feed.entries]
    # Top up with finance-focused news so the finance bucket is not starved.
    for query in FINANCE_NEWS_QUERIES:
        for e in feedparser.parse(NEWS_RSS.format(query=_quote(query))).entries:
            docs.append(_record(e.get("title", ""), e.get("summary", ""),
                                e.get("source", {}).get("title", "Financial News"),
                                "finance", e.get("link", ""), e.get("published", "")))
    return _report("finance", docs)


# --- community: hacker news ----------------------------------------------
def collect_hackernews() -> list[dict]:
    docs = []
    for query in HN_QUERIES:
        for h in _get_json(HN_API.format(query=_quote(query))).get("hits", []):
            docs.append(_record(
                h.get("title") or h.get("story_title") or "",
                h.get("story_text") or h.get("comment_text") or h.get("title") or "",
                "Hacker News", "community",
                h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                h.get("created_at", "")))
    return _report("community(hackernews)", docs)


# --- community: stack overflow -------------------------------------------
def collect_stackexchange() -> list[dict]:
    docs = []
    for query in STACKEX_QUERIES:
        for it in _get_json(STACKEX_API.format(query=_quote(query))).get("items", []):
            docs.append(_record(it.get("title", ""), it.get("body", "") or it.get("title", ""),
                                "Stack Overflow", "community",
                                it.get("link", ""), str(it.get("creation_date", ""))))
    return _report("community(stackexchange)", docs)


# --- research: arxiv -----------------------------------------------------
def collect_arxiv() -> list[dict]:
    docs = []
    for query in ARXIV_QUERIES:
        for e in feedparser.parse(ARXIV_API.format(query=_quote(query))).entries:
            docs.append(_record(e.get("title", ""), e.get("summary", ""), "arXiv",
                                "research", e.get("link", ""), e.get("published", "")))
    return _report("research(arxiv)", docs)


# --- research: openalex --------------------------------------------------
def _openalex_abstract(inverted: dict | None) -> str:
    """Rebuild plain text from OpenAlex's abstract_inverted_index ({word: [positions]})."""
    if not inverted:
        return ""
    words = [""] * (max(p for ps in inverted.values() for p in ps) + 1)
    for word, positions in inverted.items():
        for p in positions:
            words[p] = word
    return " ".join(w for w in words if w)


def collect_openalex() -> list[dict]:
    docs = []
    for query in OPENALEX_QUERIES:
        for w in _get_json(OPENALEX_API.format(query=_quote(query))).get("results", []):
            title = w.get("title") or ""
            docs.append(_record(title, _openalex_abstract(w.get("abstract_inverted_index")) or title,
                                "OpenAlex", "research",
                                w.get("doi") or w.get("id", ""), w.get("publication_date", "")))
    return _report("research(openalex)", docs)


# --- run all -------------------------------------------------------------
def collect_all() -> list[dict]:
    """Collect every source, save each type to data/raw/, return the merged list."""
    by_type = {
        "news": collect_news(),
        "finance": collect_finance(),
        "community": collect_hackernews() + collect_stackexchange(),
        "research": collect_arxiv() + collect_openalex(),
    }
    for name, docs in by_type.items():
        save_json(docs, config.RAW_DIR / f"{name}.json")

    all_docs = [d for docs in by_type.values() for d in docs]
    save_json({"collected_at": now_iso(), "total": len(all_docs), "documents": all_docs},
              config.RAW_DIR / "all_raw.json")

    print(f"[collector] TOTAL collected: {len(all_docs)} (target >= {MIN_DOCS})")
    if len(all_docs) < MIN_DOCS:
        print("[collector] WARNING: below the 100-document minimum — add more queries above")
    return all_docs


if __name__ == "__main__":
    collect_all()
