"""
Task 5 + 6 — AI CEO Agent and Evidence-Based Recommendations.

This is the reasoning layer. It does NOT retrieve or summarise for its own sake;
it converts the strategic signals from the intelligence engine into prioritised,
evidence-backed executive recommendations and a CEO briefing.

Each recommendation follows the PDF Task 6 schema:
    recommendation, supporting_evidence, expected_impact, risk_assessment, priority

The CEO briefing (PDF Section 7) answers:
    What happened?  /  Why does it matter?  /  What should management do next?

Reasoning engine: Mistral-7B-Instruct via the free HF Hub endpoint (see utils.get_llm).
"""

import re

from src import config
from src.utils import ask_llm

_IMPACT_RANK = {"High": 3, "Medium": 2, "Low": 1}


def _field(text: str, label: str, default: str = "") -> str:
    """Extract 'LABEL: value' from an LLM response."""
    m = re.search(rf"{label}\s*:\s*(.+)", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else default


def _to_list(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[;,]", text) if p.strip()]


def _signal_to_recommendation(signal: dict, kind: str) -> dict:
    """Ask the LLM to turn one opportunity/risk into an executive recommendation."""
    level = signal.get("impact") or signal.get("severity") or "Medium"
    prompt = (
        f"You are advising the CEO of {config.COMPANY}.\n"
        f"Strategic {kind}: {signal['title']}\n"
        f"Write a recommendation in EXACTLY this format:\n"
        f"RECOMMENDATION: <one decisive action>\n"
        f"IMPACT: <2-3 expected business impacts, comma separated>\n"
        f"RISK: <2-3 risks of acting, comma separated>\n"
    )
    resp = ask_llm(prompt)
    return {
        "recommendation": _field(resp, "RECOMMENDATION", default=f"Act on: {signal['title']}"),
        "priority": level if level in _IMPACT_RANK else "Medium",
        "expected_impact": _to_list(_field(resp, "IMPACT")) or ["Revenue growth", "Market differentiation"],
        "risk_assessment": _to_list(_field(resp, "RISK")) or ["Financial risk", "Execution risk"],
        "supporting_evidence": signal.get("evidence", []),
        "confidence": signal.get("confidence", 0.0),
        "source_signal": kind,
    }


def generate_recommendations(intel: dict, max_recs: int = 5) -> list[dict]:
    """Build evidence-based recommendations from the strongest opportunities + risks."""
    opportunities = sorted(
        intel.get("opportunities", []),
        key=lambda s: (_IMPACT_RANK.get(s.get("impact"), 0), s.get("confidence", 0)),
        reverse=True,
    )
    risks = sorted(
        intel.get("risks", []),
        key=lambda s: (_IMPACT_RANK.get(s.get("severity"), 0), s.get("confidence", 0)),
        reverse=True,
    )

    # Interleave opportunities and risk-mitigations, longer list tops up the tail.
    signals = [(s, kind) for pair in zip(opportunities, risks)
               for s, kind in ((pair[0], "opportunity"), (pair[1], "risk"))]
    signals += [(o, "opportunity") for o in opportunities[len(risks):]]
    signals += [(r, "risk") for r in risks[len(opportunities):]]

    # Cap to max_recs BEFORE calling the LLM — one generation per kept recommendation.
    return [_signal_to_recommendation(s, kind) for s, kind in signals[:max_recs]]


def generate_briefing(intel: dict, sentiment: dict) -> str:
    """CEO briefing answering the three PDF Section-7 questions."""
    opps = "; ".join(o["title"] for o in intel.get("opportunities", [])[:3])
    risks = "; ".join(r["title"] for r in intel.get("risks", [])[:3])
    trends = "; ".join(t["title"] for t in intel.get("trends", [])[:3])
    prompt = (
        f"You are the AI strategic advisor to the CEO of {config.COMPANY}.\n"
        f"Top opportunities: {opps}\n"
        f"Top risks: {risks}\n"
        f"Key trends: {trends}\n"
        f"News sentiment: {sentiment.get('news_sentiment')}, "
        f"public sentiment: {sentiment.get('public_sentiment')}.\n\n"
        f"Write a concise executive briefing with three short paragraphs:\n"
        f"WHAT HAPPENED:\nWHY IT MATTERS:\nWHAT TO DO NEXT:\n"
    )
    return ask_llm(prompt)


if __name__ == "__main__":
    from src.classical_agent import corpus_sentiment
    from src.intelligence_engine import run as run_intel
    from src.preprocess import load_corpus

    intel = run_intel()
    for r in generate_recommendations(intel):
        print(f"- [{r['priority']}] {r['recommendation']}")
    print("\n" + generate_briefing(intel, corpus_sentiment(load_corpus())))
