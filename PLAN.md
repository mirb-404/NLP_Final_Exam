# AI CEO — Strategic Intelligence Agent (Siemens)

NLP Final Exam project. Mirrors the structure/idioms of the course repo
`ChandnaSwati/ADSA-NLP-Modules` (Modules 1–11 + MiniHackathon) so every file
maps to something taught in class and can be explained in the oral exam.

## Decisions
- **Company:** Siemens (industry: Industrial Technology / Automation).
- **LLM (reasoning engine):** `mistralai/Mistral-7B-Instruct-v0.2` via the free
  Hugging Face Hub inference endpoint (`HuggingFaceHub`, pattern from
  `Module11/LangChain agent.ipynb`). Falls back to local `google/flan-t5-base`
  (`HuggingFacePipeline`, pattern from `Module11/LangGraph.ipynb`) when no token.
  → Satisfies PDF rule: **no paid commercial LLM API**.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Module 10 / MiniHackathon hint).
- **Vector store:** ChromaDB persistent (Module 10).
- **Retrieval:** Hybrid BM25 (sparse) + dense (Module 10 Task 3 fusion formula).
- **Orchestration:** LangGraph `StateGraph` (Module 11).
- **Env:** `uv` only (venv + deps live in `.venv`).

## PDF task → file → course source
| PDF task | File | Course source |
|---|---|---|
| 1 Live data collection (≥100 docs, ≥3 sources) | `src/collector.py` | *new* (repo ships CSVs, has no scraper) |
| 2 Knowledge repository | `src/knowledge_base.py` | Module 10 (Chroma) |
| 3 Information processing (clean/dedup/extract/embed) | `src/preprocess.py` | Module 2, 3 |
| 4 Strategic Intelligence Engine (opp/risk/trend) | `src/intelligence_engine.py` | Module 6, 9, 10 |
| 5 AI CEO Agent (reason/prioritize/recommend) | `src/ceo_agent.py` | Module 11 |
| 6 Evidence-based recommendations (verify/confidence) | `src/verifier_agent.py` | MiniHackathon verifier |
| (retrieval) | `src/retriever_hybrid.py` | Module 10 Task 3, MiniHackathon |
| (classical NLP: NER/sentiment/TF-IDF) | `src/classical_agent.py` | Module 2,3,9 + MiniHackathon |
| Orchestration pipeline | `src/orchestrator.py` | Module 11 LangGraph |

## Necessary deviation from repo
The repo modules consume *provided* CSV datasets — there is no collection code.
PDF Task 1 *requires* live collection, so `collector.py` adds 3 standard libs not
in the repo `requirements.txt`: `feedparser`, `requests`, `beautifulsoup4`
(+ `python-dotenv` for the HF token). Everything else stays inside the course stack.

## Build order (backend first; UI later)
1. `uv init` + deps ✅
2. `config.py`, `utils.py`
3. `collector.py`        → `data/raw/*.json`
4. `preprocess.py`       → `data/corpus.csv`  (≥100 clean, deduped docs)
5. `knowledge_base.py`   → `data/chroma/` persistent index
6. `retriever_hybrid.py` → top-k hybrid retrieval
7. `classical_agent.py`  → NER, sentiment, TF-IDF keywords
8. `intelligence_engine.py` → opportunities / risks / trends
9. `ceo_agent.py`        → evidence-based recommendations + CEO briefing
10. `verifier_agent.py`  → confidence scoring of each claim
11. `orchestrator.py`    → LangGraph pipeline → `results/*.json`
12. **LATER:** `app/` Streamlit dashboard (7 PDF sections)

## Outputs (`results/`)
- `recommendations.json` — recs with evidence/impact/risk/priority/confidence
- `intelligence.json` — opportunities, risks, trends
- `metrics.json` — verifier confidence/precision
- `trace.json` — LangGraph execution trace
- `ceo_briefing.txt` — what happened / why it matters / what to do next
