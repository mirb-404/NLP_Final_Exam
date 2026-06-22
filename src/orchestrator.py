"""
Task 5 — Orchestration with LangGraph (Module 11 pattern).

Wires every agent into two LangGraph pipelines:

    ingest:   collect -> process -> index -> END
    analyze:  analyze -> intelligence -> recommend -> verify -> brief -> END

State flows through a typed GraphState; each node returns partial updates
(exactly like Module11/LangGraph.ipynb). The full execution trace is written to
results/trace.json, and the assembled dashboard payload to results/dashboard_data.json.
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


# ----------------------------------------------------------------------------
# Shared state (Module 11: TypedDict with total=False)
# ----------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    docs: int
    sentiment: dict
    intel: dict
    recommendations: list
    metrics: dict
    briefing: str
    trace: list


def _trace(state: GraphState, node: str, info: dict) -> list:
    """Append one step to the running execution trace."""
    return state.get("trace", []) + [{"node": node, "time": now_iso(), "info": info}]


# ----------------------------------------------------------------------------
# Nodes — one per agent
# ----------------------------------------------------------------------------
def node_collect(state: GraphState) -> GraphState:
    docs = collect_all()
    return {"docs": len(docs), "trace": _trace(state, "collect", {"collected": len(docs)})}


def node_process(state: GraphState) -> GraphState:
    df = build_corpus()
    return {"docs": len(df), "trace": _trace(state, "process", {"clean_docs": len(df)})}


def node_index(state: GraphState) -> GraphState:
    build_index(load_corpus())
    return {"trace": _trace(state, "index", {"indexed": kb_count()})}


def node_analyze(state: GraphState) -> GraphState:
    sentiment = corpus_sentiment(load_corpus())
    return {"sentiment": sentiment, "trace": _trace(state, "analyze", sentiment)}


def node_intelligence(state: GraphState) -> GraphState:
    intel = run_intelligence()
    info = {k: len(intel[k]) for k in ("opportunities", "risks", "trends")}
    return {"intel": intel, "trace": _trace(state, "intelligence", info)}


def node_recommend(state: GraphState) -> GraphState:
    recs = generate_recommendations(state["intel"])
    return {"recommendations": recs, "trace": _trace(state, "recommend", {"n": len(recs)})}


def node_verify(state: GraphState) -> GraphState:
    recs, metrics = verify_recommendations(state["recommendations"])
    return {"recommendations": recs, "metrics": metrics,
            "trace": _trace(state, "verify", metrics)}


def node_brief(state: GraphState) -> GraphState:
    briefing = generate_briefing(state["intel"], state.get("sentiment", {}))
    return {"briefing": briefing, "trace": _trace(state, "brief", {"chars": len(briefing)})}


# ----------------------------------------------------------------------------
# Two graphs: ingest (slow, rare) and analyze (fast, repeatable).
# Splitting them means iterating on the analysis never re-collects or re-embeds.
# ----------------------------------------------------------------------------
def build_ingest_graph():
    """collect -> process -> index. Run when you want fresh data."""
    g = StateGraph(GraphState)
    g.add_node("collect", node_collect)
    g.add_node("process", node_process)
    g.add_node("index", node_index)
    g.set_entry_point("collect")
    g.add_edge("collect", "process")
    g.add_edge("process", "index")
    g.add_edge("index", END)
    return g.compile()


def build_analyze_graph():
    """analyze -> intelligence -> recommend -> verify -> brief. Reuses the stored index."""
    g = StateGraph(GraphState)
    g.add_node("analyze", node_analyze)
    g.add_node("intelligence", node_intelligence)
    g.add_node("recommend", node_recommend)
    g.add_node("verify", node_verify)
    g.add_node("brief", node_brief)
    g.set_entry_point("analyze")
    g.add_edge("analyze", "intelligence")
    g.add_edge("intelligence", "recommend")
    g.add_edge("recommend", "verify")
    g.add_edge("verify", "brief")
    g.add_edge("brief", END)
    return g.compile()


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


# Words that mark a headline as an official company action, not market chatter
# (analyst notes, "stock up 3%", fund-buys-shares, etc.). Drives Section 2.
_ANNOUNCE_RE = re.compile(
    r"\b(announces?|announced|unveils?|unveiled|launch(?:es|ed)?|introduc(?:es|ed)|"
    r"recall(?:s|ed)?|deliver(?:s|ies|ed)|earnings|results|opens?|opened|expand(?:s|ed)?|"
    r"partner(?:s|ship)?|invest(?:s|ment|ed)?|robotaxi|gigafactory|price cut|unveiling)\b",
    re.IGNORECASE,
)


def _company_announcements(n: int = 6) -> list[dict]:
    """Important company announcements (Section 2). Heuristic: recent news/finance
    headlines describing an official Tesla action (launch, recall, earnings, …),
    deduplicated by title."""
    df = load_corpus()
    pool = df[df["source_type"].isin(["news", "finance"])]
    out, seen = [], set()
    for r in pool.itertuples():
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
    """How many documents mention each top keyword — the 'signal strength' behind a
    trend, used for the Trends bar chart visualisation."""
    text = (df["title"].fillna("") + " " + df["text"].fillna("")).str.lower()
    signals = []
    for kw in keywords[:top]:
        mentions = int(text.str.contains(re.escape(kw.lower()), regex=True).sum())
        if mentions:
            signals.append({"keyword": kw, "mentions": mentions})
    return sorted(signals, key=lambda s: s["mentions"], reverse=True)


def _save_outputs(state: GraphState) -> None:
    """Persist all deliverables to results/, including the section-aligned dashboard payload."""
    save_json(state["recommendations"], config.RESULTS_DIR / "recommendations.json")
    save_json(state["trace"], config.RESULTS_DIR / "trace.json")
    save_text(state.get("briefing", ""), config.RESULTS_DIR / "ceo_briefing.txt")

    intel = state.get("intel", {})
    briefing = state.get("briefing", "")
    df = load_corpus()
    # Keys map 1:1 to the PDF dashboard sections so the frontend is a thin renderer.
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
        "sentiment": state.get("sentiment", {}),                      # Section 5 (incl. trend)
        "recommendations": state.get("recommendations", []),          # Section 6 (incl. risk_level)
        "briefing": {"raw": briefing, **_briefing_sections(briefing)},  # Section 7
        "metrics": state.get("metrics", {}),
    }
    save_json(dashboard, config.RESULTS_DIR / "dashboard_data.json")
    print(f"[orchestrator] outputs written to {config.RESULTS_DIR}")


def run_ingest() -> GraphState:
    """Refresh the knowledge base only (collect -> process -> index)."""
    return build_ingest_graph().invoke({}, {"recursion_limit": 10})


def run_analyze() -> GraphState:
    """Analyze the stored index and write all results. Fast; assumes a prior ingest."""
    state = build_analyze_graph().invoke({}, {"recursion_limit": 10})
    _save_outputs(state)
    return state


def run_pipeline() -> GraphState:
    """Full run: refresh data, then analyze."""
    run_ingest()
    return run_analyze()


if __name__ == "__main__":
    state = run_pipeline()
    print("\n=== CEO BRIEFING ===\n")
    print(state.get("briefing", ""))
