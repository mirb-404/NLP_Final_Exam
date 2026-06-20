"""
Entry point for the AI CEO backend.

Usage:
    uv run python main.py            # full pipeline (ingest + analyze)
    uv run python main.py ingest     # refresh data only: collect + corpus + index
    uv run python main.py analyze    # fast: reuse stored index -> results/  (no re-collect)
    uv run python main.py ask "..."  # tool-calling CEO agent answers one question (LangGraph)
    uv run python main.py collect    # only Task 1 (live data collection)
    uv run python main.py corpus     # only Task 3 (build clean corpus)
    uv run python main.py index      # only Task 2 (build Chroma index)

The dashboard (Streamlit) is added later and reads results/dashboard_data.json.
"""

import sys

from dotenv import load_dotenv

load_dotenv()  # read HUGGINGFACEHUB_API_TOKEN from .env


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"

    if cmd == "collect":
        from src.collector import collect_all
        collect_all()
    elif cmd == "corpus":
        from src.preprocess import build_corpus
        build_corpus()
    elif cmd == "index":
        from src.knowledge_base import build_index
        from src.preprocess import load_corpus
        build_index(load_corpus())
    elif cmd == "ingest":
        from src.orchestrator import run_ingest
        run_ingest()
    elif cmd == "ask":
        from src.agent import ask_ceo
        question = " ".join(sys.argv[2:])
        print(ask_ceo(question or "If you were the CEO today, what would you do next and why?"))
    else:  # "analyze" (fast) or "run"/default (full)
        from src.orchestrator import run_analyze, run_pipeline
        state = run_analyze() if cmd == "analyze" else run_pipeline()
        print("\n=== CEO BRIEFING ===\n")
        print(state.get("briefing", ""))


if __name__ == "__main__":
    main()
