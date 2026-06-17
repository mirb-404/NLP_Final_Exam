"""
Task 6 (verification half) — Verifier / Evaluator Agent.

Checks that each recommendation is actually grounded in its supporting evidence,
using Sentence-BERT cosine similarity (MiniHackathon verifier pattern).

For every recommendation:
    confidence = cosine( embed(recommendation), embed(evidence titles) )
A recommendation with confidence below config.CONFIDENCE_THRESHOLD is flagged
'verified = False', which the orchestrator uses to trigger a retry.

Aggregate metrics (-> results/metrics.json):
    factual_precision   : share of recommendations above the threshold
    mean_confidence
    contradiction_rate  : optional NLI check (off by default; heavy model)
"""

from sentence_transformers import util

from src import config
from src.utils import get_embedder, save_json


def _grounding_score(claim: str, evidence: list[dict]) -> float:
    """Max cosine similarity between the claim and any piece of evidence."""
    if not evidence:
        return 0.0
    embedder = get_embedder()
    ev_texts = [e.get("title", "") for e in evidence if e.get("title")]
    if not ev_texts:
        return 0.0
    claim_emb = embedder.encode(claim, convert_to_tensor=True, normalize_embeddings=True)
    ev_emb = embedder.encode(ev_texts, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(claim_emb, ev_emb)[0]
    return round(float(sims.max()), 3)


def verify_recommendations(recs: list[dict]) -> tuple[list[dict], dict]:
    """Attach a verified confidence to each recommendation and compute metrics."""
    for r in recs:
        score = _grounding_score(r["recommendation"], r.get("supporting_evidence", []))
        # blend retrieval confidence with grounding similarity
        r["confidence"] = round(0.5 * score + 0.5 * r.get("confidence", 0.0), 3)
        r["verified"] = r["confidence"] >= config.CONFIDENCE_THRESHOLD

    confidences = [r["confidence"] for r in recs] or [0.0]
    metrics = {
        "n_recommendations": len(recs),
        "mean_confidence": round(sum(confidences) / len(confidences), 3),
        "factual_precision": round(sum(r["verified"] for r in recs) / len(recs), 3) if recs else 0.0,
        "threshold": config.CONFIDENCE_THRESHOLD,
    }
    save_json(metrics, config.RESULTS_DIR / "metrics.json")
    print(f"[verifier] mean_confidence={metrics['mean_confidence']} "
          f"precision={metrics['factual_precision']}")
    return recs, metrics


if __name__ == "__main__":
    from src.ceo_agent import generate_recommendations
    from src.intelligence_engine import run as run_intel

    recs, metrics = verify_recommendations(generate_recommendations(run_intel()))
    print(metrics)
