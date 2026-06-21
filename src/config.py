"""
Central configuration for the AI CEO Strategic Intelligence Agent.

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
# Task 1 — Live data sources
# Endpoints + queries live in collector.py (one home for everything about
# collection). The company identity below (COMPANY / TICKER) drives them.
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Models (PDF "Mandatory Components" — open-source / freely accessible only)
# ----------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"               # Module 10 embedding model
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
CONFIDENCE_THRESHOLD = 0.7  # retry a recommendation if verifier confidence is below this

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
