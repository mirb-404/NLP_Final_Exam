"""
Entry point for the AI CEO backend.

Usage:
    uv run python main.py            # run the full LangGraph pipeline
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
    else:
        from src.orchestrator import run_pipeline
        state = run_pipeline()
        print("\n=== CEO BRIEFING ===\n")
        print(state.get("briefing", ""))


if __name__ == "__main__":
    main()
