"""
Central configuration for the AI CEO Strategic Intelligence Agent.

Everything that changes between runs (company, sources, model ids, thresholds,
file paths) lives here so the rest of the code never hard-codes values.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Company under analysis (Step 1 of the PDF)
# ----------------------------------------------------------------------------
COMPANY = "Siemens"
INDUSTRY = "Industrial Technology / Automation"
TICKER = "SIEGY"  # Siemens ADR on the US market (used for finance news feed)

# Competitors we also watch (used by the intelligence engine for "competitor activity")
COMPETITORS = [
    "ABB",
    "Schneider Electric",
    "General Electric",
    "Honeywell",
    "Rockwell Automation",
    "Bosch",
]

# ----------------------------------------------------------------------------
# Task 1 — Live data sources (>= 3 independent public sources)
# Every source is free and needs no paid key.
# ----------------------------------------------------------------------------
# 1) NEWS  — Google News RSS (aggregates financial / industry / tech news)
NEWS_RSS_QUERIES = [
    "Siemens",
    "Siemens Energy",
    "Siemens Healthineers",
    "Siemens automation",
    "Siemens digital industries",
]
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# 2) FINANCE — Yahoo Finance headline RSS for the Siemens ticker
YAHOO_FINANCE_RSS = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
)

# 3) COMMUNITY — Hacker News (Algolia public API, no auth, very reliable)
HN_SEARCH_JSON = "https://hn.algolia.com/api/v1/search?query={query}&tags=story&hitsPerPage=100"
HN_QUERIES = ["Siemens", "Siemens automation", "Siemens industrial"]

# 3b) COMMUNITY (best effort) — Reddit public search JSON.
# Works from a normal home IP but is often blocked (HTTP 403) from data-centre IPs,
# so it is optional: if it fails, Hacker News still provides the community source.
REDDIT_SEARCH_JSON = "https://www.reddit.com/search.json?q={query}&sort=new&limit=100"
REDDIT_QUERIES = ["Siemens", "Siemens automation", "Siemens PLC"]

# A descriptive User-Agent is required or these endpoints return HTTP 429/403.
HTTP_HEADERS = {"User-Agent": "ai-ceo-research-agent/1.0 (educational NLP project)"}

# Minimum collection target (PDF Task 1)
MIN_DOCS = 100

# ----------------------------------------------------------------------------
# Models (PDF "Mandatory Components" — open-source / freely accessible only)
# ----------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"               # Module 10 embedding model
# Primary reasoning LLM — served free via HF Inference Providers (chat_completion).
# Llama-3.1-8B-Instruct is one of the PDF-recommended models and follows the
# structured prompts cleanly. (Old serverless Mistral-7B-v0.2 is no longer hosted.)
LLM_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
LLM_FALLBACK = "Qwen/Qwen2.5-0.5B-Instruct"        # local fallback (no token / offline)
NLI_MODEL = "facebook/bart-large-mnli"             # verifier (optional contradiction check)

# ----------------------------------------------------------------------------
# Retrieval settings (Module 10 hybrid)
# ----------------------------------------------------------------------------
HYBRID_ALPHA = 0.5   # score = alpha*dense + (1-alpha)*sparse
TOP_K = 5            # documents returned per query
CHUNK_PREVIEW = 400  # characters of doc text shown as evidence

# ----------------------------------------------------------------------------
# Orchestration (Module 11)
# ----------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.7  # retry a recommendation if verifier confidence is below this

# ----------------------------------------------------------------------------
# Strategic questions the engine answers (from the PDF "Project Objective")
# Each maps to one strategic theme used to drive retrieval.
# ----------------------------------------------------------------------------
ENGINE_QUERIES = {
    "opportunities": "What are the major growth opportunities, new markets, "
                     "partnerships and emerging technologies for Siemens?",
    "risks": "What are the biggest risks, competitive threats, regulatory issues, "
             "supply chain problems and negative sentiment facing Siemens?",
    "trends": "What technology trends, industry developments and customer behaviour "
              "shifts should Siemens management monitor?",
    "competitors": "What are Siemens competitors such as ABB, Schneider Electric and "
                   "GE doing recently?",
}

# ----------------------------------------------------------------------------
# File paths (everything lives under the project root)
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CORPUS_CSV = DATA_DIR / "corpus.csv"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "siemens_docs"
RESULTS_DIR = ROOT_DIR / "results"

# Create the folders on import so no module has to worry about it.
for _d in (DATA_DIR, RAW_DIR, CHROMA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
