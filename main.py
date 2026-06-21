"""
AI CEO — Strategic Intelligence Agent  (main entry point).

main.py IS the agent graph: a LangGraph ReAct loop where the DataLab Mistral
decides which tool to call, fetches real evidence, and answers. Every capability
is a tool (src/tools.py); collection, retrieval, intelligence, etc. live in their
own modules. The graph is the brain; the modules are the hands.

    python main.py                     # agent answers the default CEO question
    python main.py ask "<question>"    # agent answers your question
    python main.py chat                # interactive: type questions in a loop
    python main.py ingest              # refresh data: collect -> corpus -> Chroma index
    python main.py report              # deterministic pipeline -> results/*.json deliverables
    python main.py collect|corpus|index   # run a single ingest stage

Tool calls are printed live so you can watch the agent fetch evidence.
"""

import re
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from src import config
from src.tools import TOOLS
from src.utils import ask_llm

load_dotenv()  # MODEL_SERVER_URL / token from .env

# ----------------------------------------------------------------------------
# THE AGENT GRAPH — LangGraph ReAct loop (Module 11)
#
#     reason --(Action?)--> act --> reason --> ... --> Final Answer (END)
#
# The model_server is a plain text endpoint (no native function calling), so the
# agent uses the ReAct pattern: the LLM writes `Action: <tool>` / `Action Input:`,
# we run that tool and feed back an `Observation:`, and loop until `Final Answer:`.
# ----------------------------------------------------------------------------
MAX_STEPS = 6  # tool calls before the agent must answer

_TOOLS_BY_NAME = {t.name: t for t in TOOLS}
_TOOL_DESCS = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in TOOLS)
_ACTION_RE = re.compile(r"Action:\s*([^\n]+)\n+Action Input:\s*([^\n]*)", re.IGNORECASE)

SYSTEM = (
    f"You are the AI strategic advisor to the CEO of {config.COMPANY}. "
    "Give concrete, prioritised, evidence-based advice citing the [src-#] markers from tools.\n\n"
    f"Available tools:\n{_TOOL_DESCS}\n\n"
    "Reason step by step in EXACTLY this format:\n"
    "Thought: <reasoning>\nAction: <one tool name>\nAction Input: <input, or NONE>\n"
    "(You then receive an `Observation:`.) When you have enough evidence, output:\n"
    "Final Answer: <the executive answer>"
)


class AgentState(TypedDict, total=False):
    question: str
    scratchpad: str   # running Thought/Action/Observation transcript
    pending: str      # the model's latest output (routed on)
    steps: int


def reason_node(state: AgentState) -> AgentState:
    """Ask the LLM for the next Thought/Action (or the Final Answer)."""
    prompt = f"{SYSTEM}\n\nQuestion: {state['question']}\n{state.get('scratchpad', '')}"
    text = ask_llm(prompt).split("Observation:")[0]   # never let the model invent an Observation
    m = _ACTION_RE.search(text)
    if m:
        text = text[: m.end()]   # keep only the first Thought + Action + Action Input; we run the tool
    text = text.strip()
    return {
        "pending": text,
        "scratchpad": state.get("scratchpad", "") + "\n" + text + "\n",
        "steps": state.get("steps", 0) + 1,
    }


def act_node(state: AgentState) -> AgentState:
    """Run the tool named in the latest Action and append its Observation (printed live)."""
    m = _ACTION_RE.search(state["pending"])
    name = m.group(1).strip().strip("`").split()[0]
    arg = m.group(2).strip()

    tool = _TOOLS_BY_NAME.get(name)
    print(f"  calling tool: {name}({arg!r})")
    if tool is None:
        obs = f"Unknown tool '{name}'. Choose one of: {list(_TOOLS_BY_NAME)}"
    else:
        try:
            obs = tool.invoke({"query": arg} if "query" in tool.args else {})
        except Exception as exc:
            obs = f"Tool error: {exc}"
    print(f"     -> {obs[:140].replace(chr(10), ' ')} ...")
    return {"scratchpad": state["scratchpad"] + f"\nObservation: {obs}\n"}


def _route(state: AgentState) -> str:
    """Stop on a Final Answer or step budget; otherwise run the requested tool."""
    if "final answer:" in state["pending"].lower() or state.get("steps", 0) >= MAX_STEPS:
        return "end"
    return "act" if _ACTION_RE.search(state["pending"]) else "end"


def build_agent():
    """Compile the ReAct graph once:  reason --(action?)--> act --> reason --> ... --> END."""
    g = StateGraph(AgentState)
    g.add_node("reason", reason_node)
    g.add_node("act", act_node)
    g.set_entry_point("reason")
    g.add_conditional_edges("reason", _route, {"act": "act", "end": END})
    g.add_edge("act", "reason")
    return g.compile()


def ask_ceo(question: str) -> str:
    """Run the agent on one strategic question and return its final answer."""
    state = build_agent().invoke(
        {"question": question, "scratchpad": "", "steps": 0},
        {"recursion_limit": 2 * MAX_STEPS + 2},
    )
    pad = state["scratchpad"]
    m = re.search(r"Final Answer:\s*(.+)", pad, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Ran out of steps without concluding -> force one clean synthesis from the evidence gathered.
    final = ask_llm(f"{SYSTEM}\n\nQuestion: {question}\n{pad}\n\n"
                    "You now have enough evidence. Reply with ONLY:\nFinal Answer:")
    return final.split("Final Answer:")[-1].strip()


def strategic_options(question: str, answer: str) -> dict:
    """Turn the agent's answer into a decision menu: strategic moves (with upside) and
    sharper follow-up questions, so the CEO sees how much further they can push."""
    prompt = (
        f"You are advising the CEO of {config.COMPANY}.\n"
        f"Question: {question}\nAnswer so far: {answer}\n\n"
        "Give the CEO a decision menu. Use EXACTLY this format, nothing else:\n"
        "OPTION: <a bold strategic move> | <the upside: how much better this could make the company>\n"
        "OPTION: <a different move> | <upside>\n"
        "OPTION: <a different move> | <upside>\n"
        "FOLLOWUP: <a sharper question the CEO should ask next>\n"
        "FOLLOWUP: <another follow-up question>\n"
        "FOLLOWUP: <another follow-up question>\n"
    )
    text = ask_llm(prompt)
    # capture "move | upside" but stop the upside at the next | or newline so a
    # same-line FOLLOWUP never bleeds into it; pull FOLLOWUPs separately.
    options = [f"{p.strip()} | {u.strip()}"
               for p, u in re.findall(r"OPTION:\s*([^|\n]+)\|\s*([^|\n]+)", text)][:3]
    followups = [f.strip() for f in re.findall(r"FOLLOWUP:\s*([^\n|]+)", text)][:3]
    return {"options": options, "followups": followups}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ask"

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
    elif cmd == "report":
        from src.orchestrator import run_analyze
        state = run_analyze()
        print("\n=== CEO BRIEFING ===\n" + state.get("briefing", ""))
    elif cmd == "chat":
        print(f"AI CEO agent for {config.COMPANY}. Ask a question (or 'exit').")
        while True:
            q = input("\n> ").strip()
            if q.lower() in ("exit", "quit", "q", ""):
                break
            print("\n" + ask_ceo(q))
    else:  # "ask" or no command -> run the agent
        question = " ".join(sys.argv[2:]) if cmd == "ask" else ""
        question = question or f"If you were the CEO of {config.COMPANY} today, what would you do next and why?"
        print(f"\n=== AI CEO AGENT ===\nQ: {question}\n")
        print("\nANSWER:\n" + ask_ceo(question))


if __name__ == "__main__":
    main()
