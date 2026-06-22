"""
Central configuration for the AI Strategy Consultant Strategic Intelligence Agent.

Everything that changes between runs (company, sources, model ids, thresholds,
file paths) lives here so the rest of the code never hard-codes values.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Company under analysis (Step 1 of the PDF)
# ----------------------------------------------------------------------------
COMPANY = "Tesla"
INDUSTRY = "Electric Vehicles / Clean Energy"
TICKER = "TSLA"  # Tesla Inc, primary US listing (used for finance news feed)

# "tesla" is also a physics unit / a person (Nikola Tesla), so the relevance filter
# matches these aliases as WHOLE WORDS. Brand/product names anchor the company.
COMPANY_ALIASES = [
    "Tesla",
    "TSLA",
    "Elon Musk",
    "Model 3",
    "Model Y",
    "Model S",
    "Cybertruck",
    "Gigafactory",
    "Powerwall",
]

# Competitors we also watch (used by the intelligence engine for "competitor activity")
COMPETITORS = [
    "BYD",
    "Rivian",
    "Lucid",
    "Ford",
    "General Motors",
    "Volkswagen",
    "NIO",
]

# ----------------------------------------------------------------------------
# Keyword / trend-signal noise filter
# ----------------------------------------------------------------------------
# Terms that dominate a single-company corpus but are NOT trends: news-source /
# publisher names and generic finance filler. The company's own identity tokens
# (COMPANY / TICKER / COMPANY_ALIASES) are added automatically below, so "tesla",
# "tsla", "model 3" etc. never get counted as an emerging trend.
KEYWORD_STOPWORDS = {
    # publisher / source / platform names that leak in from feed metadata
    "yahoo", "finance", "news", "com", "reuters", "bloomberg", "marketbeat",
    "benzinga", "globe", "mail", "motley", "fool", "cnbc", "forbes", "insider",
    "nasdaq", "zacks", "tipranks", "barron", "seeking", "alpha", "watch",
    "stocktwits", "reddit", "twitter", "hackernews", "stackoverflow", "arxiv",
    "openalex", "tikr", "oracle",
    # generic finance / filler
    "stock", "stocks", "shares", "share", "market", "markets", "price", "prices",
    "company", "inc", "report", "reports", "says", "said", "new", "year", "years",
    "today", "week", "day", "billion", "million",
    # fund-trading boilerplate that floods finance feeds
    # ("X Capital LLC Purchases/Sells Shares of Tesla, Inc. $TSLA")
    "llc", "ltd", "lp", "capital", "partners", "ventures", "holdings", "group",
    "management", "advisors", "advisory", "asset", "investment", "investments",
    "fund", "funds", "trust", "purchases", "sells", "buys", "position", "stake",
    "takes", "makes", "increases", "decreases", "reduces", "boosts", "trims",
    "hedge", "according", "bank", "savings",
}

# Company identity is noise (it's the subject of every doc); competitors are NOT —
# a competitor surging is itself a trend worth monitoring, and has its own panel.
_KEYWORD_STOP = set(KEYWORD_STOPWORDS)
for _name in COMPANY_ALIASES + [COMPANY, TICKER]:
    _KEYWORD_STOP.update(_name.lower().split())


def keyword_is_noise(term: str) -> bool:
    """True if a keyword/phrase is domain noise — i.e. any of its tokens is the
    company itself, its ticker/products, a news source, or generic finance filler.
    Used to keep the trend-signal chart and keyword lists meaningful."""
    return any(w in _KEYWORD_STOP for w in str(term).lower().split())


# ----------------------------------------------------------------------------
# Task 1 — Live data sources
# Endpoints + queries live in collector.py (one home for everything about
# collection). The company identity below (COMPANY / TICKER) drives them.
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Models (PDF "Mandatory Components" — open-source / freely accessible only)
# ----------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"               # Module 10 embedding model
# Classical sentiment model (Module 9 transformers pipeline) — DistilBERT fine-tuned on
# SST-2, the course-reference sentiment model. It outputs POSITIVE/NEGATIVE; we map it to
# a signed -1..+1 score in classical_agent.py so news/public sentiment stays comparable.
SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
# Primary reasoning LLM: PDF-recommended Mistral-7B-Instruct, served by the local
# DataLab model_server (OpenAI-compatible). This id MUST match what your server
# advertises at GET /v1/models  (curl http://127.0.0.1:8000/v1/models).
LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_FALLBACK = "Qwen/Qwen2.5-0.5B-Instruct"        # local fallback (no token / offline)

# ----------------------------------------------------------------------------
# Retrieval settings (Module 10 hybrid)
# ----------------------------------------------------------------------------
HYBRID_ALPHA = 0.5   # score = alpha*dense + (1-alpha)*sparse
TOP_K = 5            # documents returned per query
CHUNK_PREVIEW = 400  # characters of doc text shown as evidence

# ----------------------------------------------------------------------------
# Orchestration (Module 11)
# ----------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.7  # recommendations below this are flagged unverified (factual_precision)

# ----------------------------------------------------------------------------
# Strategic questions the engine answers (from the PDF "Project Objective")
# Each maps to one strategic theme used to drive retrieval.
# ----------------------------------------------------------------------------
ENGINE_QUERIES = {
    "opportunities": "What are the major growth opportunities, new markets, "
                     "partnerships and emerging technologies for Tesla?",
    "risks": "What are the biggest risks, competitive threats, regulatory issues, "
             "supply chain problems and negative sentiment facing Tesla?",
    "trends": "What technology trends, industry developments and customer behaviour "
              "shifts should Tesla management monitor?",
    "competitors": "What are Tesla competitors such as BYD, Rivian and "
                   "Ford doing recently?",
}

# ----------------------------------------------------------------------------
# File paths (everything lives under the project root)
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CORPUS_CSV = DATA_DIR / "corpus.csv"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "tesla_docs"
RESULTS_DIR = ROOT_DIR / "results"

# Create the folders on import so no module has to worry about it.
for _d in (DATA_DIR, RAW_DIR, CHROMA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
