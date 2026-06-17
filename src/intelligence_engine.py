"""
Task 4 — Strategic Intelligence Engine.

Turns the indexed corpus into structured strategic signals:

  - opportunities : emerging tech, new markets, partnerships, products
  - risks         : competitive / regulatory / sentiment / supply-chain
  - trends        : technology + industry + customer-behaviour shifts
  - competitor_activity

For each theme we:
  1. retrieve the most relevant documents (hybrid BM25 + dense),
  2. ask the LLM to extract a short list of concrete items grounded in that
     evidence (one item per line, easy to parse),
  3. attach the supporting evidence + a confidence score derived from the
     retrieval scores.

Output -> results/intelligence.json
"""

import re

from src import config
from src.classical_agent import top_keywords
from src.preprocess import load_corpus
from src.retriever_hybrid import HybridRetriever
from src.utils import ask_llm, save_json

EVIDENCE_PER_ITEM = 3


def _evidence_block(docs: list[dict]) -> str:
    """Format retrieved docs as a numbered, citable context block."""
    lines = []
    for d in docs:
        preview = d["text"][: config.CHUNK_PREVIEW]
        lines.append(f"[src-{d['rank']}] ({d['source']}) {d['title']}. {preview}")
    return "\n".join(lines)


def _evidence_list(docs: list[dict]) -> list[dict]:
    """Compact evidence objects for the dashboard / recommendations."""
    return [
        {"ref": f"src-{d['rank']}", "title": d["title"], "source": d["source"],
         "url": d["url"], "score": d["score"]}
        for d in docs[:EVIDENCE_PER_ITEM]
    ]


def _confidence(docs: list[dict]) -> float:
    """Mean fused retrieval score of the supporting evidence (already in [0,1])."""
    top = docs[:EVIDENCE_PER_ITEM]
    return round(sum(d["score"] for d in top) / len(top), 3) if top else 0.0


def _parse_lines(raw: str) -> list[str]:
    """Pull clean items out of an LLM numbered/bulleted list."""
    items = []
    for line in raw.splitlines():
        # strip leading list markers: "1.", "1)", "[1]", "-", "*", "•"
        line = re.sub(r"^\s*(?:\[?\d+[\].)]|[-*•])\s*", "", line).strip()
        line = line.strip("*").strip()  # drop markdown bold
        # skip blank lines and lines that just echo the format template
        if len(line) <= 3:
            continue
        if re.fullmatch(r"(?:high|medium|low|\s|::|\|)+", line, flags=re.IGNORECASE):
            continue
        items.append(line)
    return items[:5]


def detect_opportunities(retriever: HybridRetriever) -> list[dict]:
    docs = retriever.retrieve(config.ENGINE_QUERIES["opportunities"])
    prompt = (
        f"You are a strategy analyst for {config.COMPANY}. Using ONLY the evidence "
        f"below, list up to 5 concrete business OPPORTUNITIES.\n"
        f"One per line, format:  <short opportunity> :: <High|Medium|Low impact>\n\n"
        f"EVIDENCE:\n{_evidence_block(docs)}\n\nOPPORTUNITIES:"
    )
    items = []
    for line in _parse_lines(ask_llm(prompt)):
        title, _, impact = line.partition("::")
        if not title.strip():
            continue  # drop malformed lines that carry no actual opportunity
        items.append({
            "title": title.strip(),
            "impact": (impact.strip() or "Medium").split()[0].capitalize(),
            "confidence": _confidence(docs),
            "evidence": _evidence_list(docs),
        })
    return items


def detect_risks(retriever: HybridRetriever) -> list[dict]:
    docs = retriever.retrieve(config.ENGINE_QUERIES["risks"])
    prompt = (
        f"You are a risk analyst for {config.COMPANY}. Using ONLY the evidence below, "
        f"list up to 5 concrete RISKS.\nOne per line, format:  "
        f"<short risk> :: <competitive|regulatory|sentiment|supply chain|financial> :: "
        f"<High|Medium|Low severity>\n\nEVIDENCE:\n{_evidence_block(docs)}\n\nRISKS:"
    )
    items = []
    for line in _parse_lines(ask_llm(prompt)):
        parts = [p.strip() for p in line.split("::")]
        if not parts[0]:
            continue  # drop malformed lines that carry no actual risk
        items.append({
            "title": parts[0],
            "category": parts[1] if len(parts) > 1 else "competitive",
            "severity": (parts[2].split()[0].capitalize() if len(parts) > 2 else "Medium"),
            "confidence": _confidence(docs),
            "evidence": _evidence_list(docs),
        })
    return items


def detect_trends(retriever: HybridRetriever, df) -> list[dict]:
    docs = retriever.retrieve(config.ENGINE_QUERIES["trends"])
    prompt = (
        f"Using ONLY the evidence below, list up to 5 emerging TRENDS "
        f"{config.COMPANY} management should monitor (technology, industry, customer "
        f"behaviour). One short trend per line.\n\n"
        f"EVIDENCE:\n{_evidence_block(docs)}\n\nTRENDS:"
    )
    keywords = top_keywords((df["title"] + " " + df["text"]).tolist())
    return [
        {"title": t, "confidence": _confidence(docs), "evidence": _evidence_list(docs),
         "keywords": keywords[:8]}
        for t in _parse_lines(ask_llm(prompt))
    ]


def competitor_activity(retriever: HybridRetriever) -> list[dict]:
    docs = retriever.retrieve(config.ENGINE_QUERIES["competitors"])
    return _evidence_list(docs)


def run() -> dict:
    df = load_corpus()
    retriever = HybridRetriever()
    intel = {
        "opportunities": detect_opportunities(retriever),
        "risks": detect_risks(retriever),
        "trends": detect_trends(retriever, df),
        "competitor_activity": competitor_activity(retriever),
        "keywords": top_keywords((df["title"] + " " + df["text"]).tolist()),
    }
    save_json(intel, config.RESULTS_DIR / "intelligence.json")
    print(f"[intelligence_engine] {len(intel['opportunities'])} opportunities, "
          f"{len(intel['risks'])} risks, {len(intel['trends'])} trends")
    return intel


if __name__ == "__main__":
    run()
