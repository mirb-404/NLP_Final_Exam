"""
Task 1 — Live Data Collection.

Automatically pulls live, public information about the company from THREE
independent sources and writes the raw documents to data/raw/*.json:

  1. NEWS      -> Google News RSS  (financial / industry / tech news)
  2. FINANCE   -> Yahoo Finance RSS (headlines for the ticker)
  3. COMMUNITY -> Reddit public search JSON (public opinion / sentiment)

Each collected record has the same shape so the rest of the pipeline is uniform:
    {id, title, text, source, source_type, url, date}

NOTE: the course repo ships ready-made CSV datasets, so it has no scraper. The
PDF requires *live* collection, so this file uses three small standard libraries
(`feedparser`, `requests`) that are not in the repo's requirements.txt.
"""

import hashlib

import feedparser
import requests

from src import config
from src.utils import clean_text, now_iso, save_json


def _make_id(url: str, title: str) -> str:
    """Stable short id from url+title (used later for de-duplication)."""
    return hashlib.md5(f"{url}|{title}".encode("utf-8")).hexdigest()[:12]


def _record(title, text, source, source_type, url, date):
    return {
        "id": _make_id(url or title, title),
        "title": clean_text(title),
        "text": clean_text(text) or clean_text(title),
        "source": source,
        "source_type": source_type,      # news | finance | community
        "url": url,
        "date": date or "",
    }


# ----------------------------------------------------------------------------
# Source 1 — Google News RSS
# ----------------------------------------------------------------------------
def collect_news() -> list[dict]:
    docs = []
    for query in config.NEWS_RSS_QUERIES:
        url = config.GOOGLE_NEWS_RSS.format(query=requests.utils.quote(query))
        feed = feedparser.parse(url)
        for entry in feed.entries:
            docs.append(
                _record(
                    title=entry.get("title", ""),
                    text=entry.get("summary", ""),
                    source=entry.get("source", {}).get("title", "Google News"),
                    source_type="news",
                    url=entry.get("link", ""),
                    date=entry.get("published", ""),
                )
            )
    print(f"[collector] news: {len(docs)} items")
    return docs


# ----------------------------------------------------------------------------
# Source 2 — Yahoo Finance RSS
# ----------------------------------------------------------------------------
def collect_finance() -> list[dict]:
    url = config.YAHOO_FINANCE_RSS.format(ticker=config.TICKER)
    feed = feedparser.parse(url)
    docs = [
        _record(
            title=entry.get("title", ""),
            text=entry.get("summary", ""),
            source="Yahoo Finance",
            source_type="finance",
            url=entry.get("link", ""),
            date=entry.get("published", ""),
        )
        for entry in feed.entries
    ]
    print(f"[collector] finance: {len(docs)} items")
    return docs


# ----------------------------------------------------------------------------
# Source 3 — Hacker News (Algolia public API)  [community / tech discussion]
# ----------------------------------------------------------------------------
def collect_hackernews() -> list[dict]:
    docs = []
    for query in config.HN_QUERIES:
        url = config.HN_SEARCH_JSON.format(query=requests.utils.quote(query))
        try:
            resp = requests.get(url, headers=config.HTTP_HEADERS, timeout=20)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
        except Exception as exc:
            print(f"[collector] hackernews query '{query}' failed: {exc}")
            continue
        for h in hits:
            docs.append(
                _record(
                    title=h.get("title") or h.get("story_title") or "",
                    text=h.get("story_text") or h.get("comment_text") or h.get("title") or "",
                    source="Hacker News",
                    source_type="community",
                    url=h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                    date=h.get("created_at", ""),
                )
            )
    print(f"[collector] community(hackernews): {len(docs)} items")
    return docs


# ----------------------------------------------------------------------------
# Source 3b — Reddit public search JSON (best effort; may be blocked)
# ----------------------------------------------------------------------------
def collect_reddit() -> list[dict]:
    docs = []
    for query in config.REDDIT_QUERIES:
        url = config.REDDIT_SEARCH_JSON.format(query=requests.utils.quote(query))
        try:
            resp = requests.get(url, headers=config.HTTP_HEADERS, timeout=20)
            resp.raise_for_status()
            children = resp.json().get("data", {}).get("children", [])
        except Exception as exc:
            print(f"[collector] reddit query '{query}' failed: {exc}")
            continue
        for child in children:
            d = child.get("data", {})
            docs.append(
                _record(
                    title=d.get("title", ""),
                    text=d.get("selftext", "") or d.get("title", ""),
                    source=f"reddit/{d.get('subreddit', '?')}",
                    source_type="community",
                    url="https://www.reddit.com" + d.get("permalink", ""),
                    date=str(d.get("created_utc", "")),
                )
            )
    print(f"[collector] community(reddit): {len(docs)} items")
    return docs


# ----------------------------------------------------------------------------
# Run all sources
# ----------------------------------------------------------------------------
def collect_all() -> list[dict]:
    """Collect from every source, save each to data/raw/, return the merged list."""
    news = collect_news()
    finance = collect_finance()
    community = collect_hackernews() + collect_reddit()  # HN reliable, Reddit best-effort

    save_json(news, config.RAW_DIR / "news.json")
    save_json(finance, config.RAW_DIR / "finance.json")
    save_json(community, config.RAW_DIR / "community.json")

    all_docs = news + finance + community
    save_json(
        {"collected_at": now_iso(), "total": len(all_docs), "documents": all_docs},
        config.RAW_DIR / "all_raw.json",
    )

    print(f"[collector] TOTAL collected: {len(all_docs)} (target >= {config.MIN_DOCS})")
    if len(all_docs) < config.MIN_DOCS:
        print("[collector] WARNING: below the 100-document minimum — add more queries in config.py")
    return all_docs


if __name__ == "__main__":
    collect_all()
