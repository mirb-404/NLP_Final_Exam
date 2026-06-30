"""
Orchestration — ONE linear LangGraph pipeline (Module 11).

    collect → process → index → analyze → intelligence → recommend → verify → brief

The whole system is just TWO graphs:
  1. this linear pipeline (below) — the deterministic deliverables, run start-to-finish;
  2. the ReAct agent loop in main.py — interactive Q&A.

Everything else (collector / preprocess / knowledge_base / retriever / classical_agent /
intelligence_engine / ceo_agent / verifier_agent) is a plain stage function that a node
calls. Each node is a one-liner; the real logic lives in those modules.
"""

import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src import config
from src.ceo_agent import generate_briefing, generate_recommendations
from src.classical_agent import corpus_sentiment
from src.collector import collect_all
from src.intelligence_engine import run as run_intelligence
from src.knowledge_base import build_index, count as kb_count
from src.preprocess import build_corpus, load_corpus
from src.utils import now_iso, save_json, save_text
from src.verifier_agent import verify_recommendations


class State(TypedDict, total=False):
    sentiment: dict
    intel: dict
    recommendations: list
    metrics: dict
    briefing: str


# ----------------------------------------------------------------------------
# Pipeline stages — one node each, each delegating to a stage module.
# ----------------------------------------------------------------------------
def collect(state: State) -> State:
    collect_all()
    return {}


def process(state: State) -> State:
    build_corpus()
    return {}


def index(state: State) -> State:
    build_index(load_corpus())
    return {}


def analyze(state: State) -> State:
    return {"sentiment": corpus_sentiment(load_corpus())}


def intelligence(state: State) -> State:
    return {"intel": run_intelligence()}


def recommend(state: State) -> State:
    return {"recommendations": generate_recommendations(state["intel"])}


def verify(state: State) -> State:
    recs, metrics = verify_recommendations(state["recommendations"])
    return {"recommendations": recs, "metrics": metrics}


def brief(state: State) -> State:
    return {"briefing": generate_briefing(state["intel"], state.get("sentiment", {}))}


INGEST = [collect, process, index]                                   # refresh the data
ANALYZE = [analyze, intelligence, recommend, verify, brief]          # reason over the index


def _run(steps: list) -> State:
    """Build and run a linear graph from an ordered list of step functions."""
    g = StateGraph(State)
    for fn in steps:
        g.add_node(fn.__name__, fn)
    g.set_entry_point(steps[0].__name__)
    for a, b in zip(steps, steps[1:]):
        g.add_edge(a.__name__, b.__name__)
    g.add_edge(steps[-1].__name__, END)
    return g.compile().invoke({}, {"recursion_limit": len(steps) + 5})


# ----------------------------------------------------------------------------
# Public API — same names the CLI and dashboard already call.
# ----------------------------------------------------------------------------
def run_ingest() -> State:
    """collect → process → index. Run when you want fresh data."""
    return _run(INGEST)


def run_analyze() -> State:
    """analyze → … → brief over the stored index, then write every result file. Fast."""
    state = _run(ANALYZE)
    _save_outputs(state)
    return state


def run_pipeline() -> State:
    """Full run: refresh data, then analyze."""
    run_ingest()
    return run_analyze()


# ----------------------------------------------------------------------------
# Dashboard payload — assemble results/dashboard_data.json (keys map 1:1 to the
# PDF dashboard sections, so the Streamlit frontend stays a thin renderer).
# ----------------------------------------------------------------------------
def _briefing_sections(text: str) -> dict:
    """Split the CEO briefing into the three PDF Section-7 questions."""
    labels = [
        ("what_happened", r"WHAT HAPPENED"),
        ("why_it_matters", r"WHY (?:IT|DOES IT) MATTER[S]?"),
        ("what_next", r"WHAT (?:TO DO NEXT|SHOULD MANAGEMENT DO NEXT)"),
    ]
    out = {}
    for i, (key, pat) in enumerate(labels):
        nxt = labels[i + 1][1] if i + 1 < len(labels) else r"\Z"
        m = re.search(rf"{pat}\s*:?\s*(.+?)(?={nxt}|\Z)", text, re.IGNORECASE | re.DOTALL)
        out[key] = m.group(1).strip() if m else ""
    return out


def _recent_news(n: int = 8) -> list[dict]:
    """A few recent news/finance headlines for the dashboard Market Intelligence panel."""
    df = load_corpus()
    news = df[df["source_type"].isin(["news", "finance"])].head(n)
    return [{"title": r.title, "source": r.source, "url": r.url, "date": str(r.date)}
            for r in news.itertuples()]


# Words that mark a headline as an official company action (not market chatter).
_ANNOUNCE_RE = re.compile(
    r"\b(announces?|announced|unveils?|unveiled|launch(?:es|ed)?|introduc(?:es|ed)|"
    r"recall(?:s|ed)?|deliver(?:s|ies|ed)|earnings|results|opens?|opened|expand(?:s|ed)?|"
    r"partner(?:s|ship)?|invest(?:s|ment|ed)?|robotaxi|gigafactory|price cut|unveiling)\b",
    re.IGNORECASE,
)


def _company_announcements(n: int = 6) -> list[dict]:
    """Important company announcements (Section 2): recent news/finance headlines describing
    an official Tesla action (launch, recall, earnings, …), deduplicated by title."""
    df = load_corpus()
    out, seen = [], set()
    for r in df[df["source_type"].isin(["news", "finance"])].itertuples():
        title = str(r.title)
        key = title.lower().strip()
        if key in seen or not _ANNOUNCE_RE.search(title):
            continue
        seen.add(key)
        out.append({"title": title, "source": r.source, "url": r.url, "date": str(r.date)})
        if len(out) >= n:
            break
    return out


def _trend_signals(df, keywords: list[str], top: int = 10) -> list[dict]:
    """How many documents mention each top keyword — the 'signal strength' behind a trend."""
    text = (df["title"].fillna("") + " " + df["text"].fillna("")).str.lower()
    signals = [{"keyword": kw, "mentions": int(text.str.contains(re.escape(kw.lower()), regex=True).sum())}
               for kw in keywords[:top]]
    return sorted((s for s in signals if s["mentions"]), key=lambda s: s["mentions"], reverse=True)


def _save_outputs(state: State) -> None:
    """Persist all deliverables to results/, including the section-aligned dashboard payload."""
    save_json(state.get("recommendations", []), config.RESULTS_DIR / "recommendations.json")
    save_text(state.get("briefing", ""), config.RESULTS_DIR / "ceo_briefing.txt")

    intel = state.get("intel", {})
    briefing = state.get("briefing", "")
    df = load_corpus()
    dashboard = {
        "company": {                                                  # Section 1
            "name": config.COMPANY,
            "industry": config.INDUSTRY,
            "n_documents": kb_count(),
            "n_sources": int(df["source_type"].nunique()),
            "source_breakdown": df["source_type"].value_counts().to_dict(),
            "last_update": now_iso(),
        },
        "market_intelligence": {                                      # Section 2
            "recent_news": _recent_news(),
            "competitor_activity": intel.get("competitor_activity", []),
            "emerging_technologies": intel.get("trends", []),
            "company_announcements": _company_announcements(),
            "keywords": intel.get("keywords", []),
        },
        "opportunities": intel.get("opportunities", []),              # Section 3
        "risks": intel.get("risks", []),                             # Section 4
        "trends": {                                                   # Task 4 — Trends
            "items": intel.get("trends", []),
            "signals": _trend_signals(df, intel.get("keywords", [])),
        },
        "sentiment": state.get("sentiment", {}),                      # Section 5
        "recommendations": state.get("recommendations", []),          # Section 6
        "briefing": {"raw": briefing, **_briefing_sections(briefing)},  # Section 7
        "metrics": state.get("metrics", {}),
    }
    save_json(dashboard, config.RESULTS_DIR / "dashboard_data.json")
    print(f"[orchestrator] outputs written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    state = run_pipeline()
    print("\n=== CEO BRIEFING ===\n")
    print(state.get("briefing", ""))
