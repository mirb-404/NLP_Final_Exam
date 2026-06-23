"""
Task 1 — Live Data Collection.

Pulls live public docs from 4 source types into data/raw/*.json, all in one uniform
shape {id, title, text, source, source_type, url, date}:
    news=Google News · finance=Yahoo Finance · community=Hacker News + Stack Overflow · research=arXiv + OpenAlex

API sources (HN / Stack Overflow / arXiv / OpenAlex) are fetched PAGES_PER_QUERY pages
deep per query, and collect_all() MERGES each run with prior runs (dedup by id) — so a
refresh ADDS new documents and grows the corpus instead of replacing it.
"""

import hashlib

import feedparser
import requests

from src import config
from src.utils import clean_text, load_json, now_iso, save_json

# --- source endpoints + queries ---------------------------------------------
HEADERS = {"User-Agent": "ai-strategy-consultant-agent/1.0 (educational NLP project)"}
TIMEOUT = 25
MIN_DOCS = 100        # PDF Task 1 minimum
PAGES_PER_QUERY = 2   # how many pages deep to fetch each API query (RSS feeds can't paginate)

# Queries say "Tesla Inc" / product names, never bare "Tesla", to keep the physics unit out.
NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
NEWS_QUERIES = ["Tesla Inc", "Tesla Model 3", "Tesla Cybertruck",
                "Tesla Elon Musk", "Tesla autopilot"]

FINANCE_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
# Yahoo's headline RSS alone is thin, so we also pull finance-focused news (tagged finance).
FINANCE_NEWS_QUERIES = ["TSLA stock", "Tesla earnings", "Tesla stock forecast",
                        "Tesla deliveries", "Tesla revenue"]

# API sources carry a page/offset placeholder filled in by the paged loops below.
HN_API = "https://hn.algolia.com/api/v1/search?query={query}&tags=story&hitsPerPage=60&page={page}"
HN_QUERIES = ["Tesla Inc", "Tesla autopilot", "Tesla FSD"]

STACKEX_API = ("https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance"
               "&q={query}&site=stackoverflow&pagesize=40&filter=withbody&page={page}")
STACKEX_QUERIES = ["Tesla API", "Tesla fleet API", "Teslamate"]

ARXIV_API = "http://export.arxiv.org/api/query?search_query=all:{query}&start={start}&max_results=30"
ARXIV_QUERIES = ["Tesla autonomous driving", "Tesla electric vehicle battery", "Tesla Motors"]

OPENALEX_API = "https://api.openalex.org/works?search={query}&per_page=30&page={page}&mailto=bhandarimirang03@gmail.com"
OPENALEX_QUERIES = ["Tesla electric vehicle", "Tesla battery technology", "Tesla autonomous driving"]


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


# --- community: hacker news (paged) --------------------------------------
def collect_hackernews() -> list[dict]:
    docs = []
    for query in HN_QUERIES:
        for page in range(PAGES_PER_QUERY):                      # Algolia pages are 0-indexed
            hits = _get_json(HN_API.format(query=_quote(query), page=page)).get("hits", [])
            if not hits:
                break                                            # no more results for this query
            for h in hits:
                docs.append(_record(
                    h.get("title") or h.get("story_title") or "",
                    h.get("story_text") or h.get("comment_text") or h.get("title") or "",
                    "Hacker News", "community",
                    h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                    h.get("created_at", "")))
    return _report("community(hackernews)", docs)


# --- community: stack overflow (paged) -----------------------------------
def collect_stackexchange() -> list[dict]:
    docs = []
    for query in STACKEX_QUERIES:
        for page in range(1, PAGES_PER_QUERY + 1):               # StackExchange pages are 1-indexed
            data = _get_json(STACKEX_API.format(query=_quote(query), page=page))
            items = data.get("items", [])
            for it in items:
                docs.append(_record(it.get("title", ""), it.get("body", "") or it.get("title", ""),
                                    "Stack Overflow", "community",
                                    it.get("link", ""), str(it.get("creation_date", ""))))
            if not items or not data.get("has_more"):
                break                                            # API says no further pages
    return _report("community(stackexchange)", docs)


# --- research: arxiv (paged) ---------------------------------------------
def collect_arxiv() -> list[dict]:
    docs = []
    for query in ARXIV_QUERIES:
        for page in range(PAGES_PER_QUERY):                      # arXiv pages via start offset
            entries = feedparser.parse(ARXIV_API.format(query=_quote(query), start=page * 30)).entries
            if not entries:
                break
            for e in entries:
                docs.append(_record(e.get("title", ""), e.get("summary", ""), "arXiv",
                                    "research", e.get("link", ""), e.get("published", "")))
    return _report("research(arxiv)", docs)


# --- research: openalex (paged) ------------------------------------------
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
        for page in range(1, PAGES_PER_QUERY + 1):               # OpenAlex pages are 1-indexed
            results = _get_json(OPENALEX_API.format(query=_quote(query), page=page)).get("results", [])
            if not results:
                break
            for w in results:
                title = w.get("title") or ""
                docs.append(_record(title, _openalex_abstract(w.get("abstract_inverted_index")) or title,
                                    "OpenAlex", "research",
                                    w.get("doi") or w.get("id", ""), w.get("publication_date", "")))
    return _report("research(openalex)", docs)


# --- run all -------------------------------------------------------------
def collect_all() -> list[dict]:
    """Collect every source and MERGE with prior runs (dedup by id) so each refresh ADDS
    new documents instead of replacing the corpus. Saves per-type files + all_raw.json."""
    fresh = (collect_news() + collect_finance()
             + collect_hackernews() + collect_stackexchange()
             + collect_arxiv() + collect_openalex())

    # accumulate: load everything seen before, then add the new ids on top (id = dedup key)
    path = config.RAW_DIR / "all_raw.json"
    merged = {}
    if path.exists():
        try:
            for d in load_json(path).get("documents", []):
                merged[d["id"]] = d
        except Exception:
            pass
    before = len(merged)
    for d in fresh:
        merged.setdefault(d["id"], d)
    all_docs = list(merged.values())
    added = len(merged) - before

    # per-type snapshots from the accumulated set (for inspection)
    for stype in ("news", "finance", "community", "research"):
        save_json([d for d in all_docs if d["source_type"] == stype],
                  config.RAW_DIR / f"{stype}.json")
    save_json({"collected_at": now_iso(), "total": len(all_docs), "documents": all_docs}, path)

    print(f"[collector] TOTAL accumulated: {len(all_docs)} (+{added} new this run, target >= {MIN_DOCS})")
    if len(all_docs) < MIN_DOCS:
        print("[collector] WARNING: below the 100-document minimum — add more queries above")
    return all_docs


if __name__ == "__main__":
    collect_all()
