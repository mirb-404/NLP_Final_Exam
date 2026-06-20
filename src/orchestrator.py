"""
Task 5 — Orchestration with LangGraph (Module 11 pattern).

Wires every agent into a single automated pipeline:

    collect -> process -> index -> analyze -> intelligence
            -> recommend -> verify --(confidence < 0.7?)--> recommend (retry once)
                                   --(ok)--> brief -> END

State flows through a typed GraphState; each node returns partial updates
(exactly like Module11/LangGraph.ipynb). The full execution trace is written to
results/trace.json, and the assembled dashboard payload to results/dashboard_data.json.
"""

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


def _save_outputs(state: GraphState) -> None:
    """Persist all deliverables to results/ (also feeds the future dashboard)."""
    save_json(state["recommendations"], config.RESULTS_DIR / "recommendations.json")
    save_json(state["trace"], config.RESULTS_DIR / "trace.json")
    save_text(state.get("briefing", ""), config.RESULTS_DIR / "ceo_briefing.txt")

    # Counts come from the persisted corpus/index, so analyze-only runs report correctly.
    n_sources = int(load_corpus()["source_type"].nunique())
    dashboard = {
        "company": {
            "name": config.COMPANY,
            "industry": config.INDUSTRY,
            "n_documents": kb_count(),
            "n_sources": n_sources,
            "last_update": now_iso(),
        },
        "sentiment": state.get("sentiment", {}),
        "intelligence": state.get("intel", {}),
        "recommendations": state.get("recommendations", []),
        "metrics": state.get("metrics", {}),
        "briefing": state.get("briefing", ""),
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
