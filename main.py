"""
main.py — Wire everything together and run the deep agent

This file is the only place that knows about ALL the other files.
It assembles the system and fires it.

Reading order for understanding:
  state.py  →  loop.py  →  tools.py  →  agent.py  →  main.py (here)

Everything flows into main.py. Nothing flows out.
"""

from state import make_state
from loop import run_agent
from tools import BASE_TOOLS, BASE_SCHEMAS
from agent import create_task_tool
from prompts import RESEARCHER_PROMPT, build_parent_prompt


def print_separator(title: str = "") -> None:
    """Print a visual separator for readable output."""
    line = "=" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(f"{line}\n")
    else:
        print(f"\n{line}\n")


def print_final_answer(state: dict) -> None:
    """Print the agent's final text response."""
    for msg in reversed(state["messages"]):
        if msg["role"] == "assistant":
            content = msg["content"]
            if isinstance(content, list):
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        print(block["text"])
                        return
            elif isinstance(content, str):
                print(content)
                return
    print("(No final response found)")


def print_state_summary(state: dict) -> None:
    """Print a summary of the final state for inspection."""
    print_separator("Final state summary")

    # Show todos
    todos = state.get("todos", [])
    if todos:
        print("TODOs:")
        for todo in todos:
            emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(
                todo.get("status", "pending"), "❓"
            )
            print(f"  {emoji} {todo['content']} ({todo['status']})")
    else:
        print("TODOs: (none)")

    print()

    # Show files
    files = state.get("files", {})
    if files:
        print("Files saved:")
        for filename, content in files.items():
            print(f"  - {filename} ({len(content)} chars)")
    else:
        print("Files: (none)")

    print()

    # Show turn count
    turns = len([m for m in state["messages"] if m["role"] == "assistant"])
    print(f"Total LLM turns: {turns}")


def main():
    # ── 1. DEFINE SUB-AGENT CONFIGURATIONS ──────────────────────────────────
    # Same structure as LangGraph version — list of dicts describing each specialist.
    # "tools" lists the names of tools this sub-agent is allowed to use.
    research_sub_agent = {
        "name": "research-agent",
        "description": "Searches the web for a specific topic. Give one topic at a time.",
        "prompt": RESEARCHER_PROMPT,
        "tools": ["web_search", "think_tool"],  # researcher gets search only
    }

    # To add a writer agent — just add another dict here:
    # writer_sub_agent = {
    #     "name": "writer-agent",
    #     "description": "Writes reports from research files.",
    #     "prompt": WRITER_PROMPT,
    #     "tools": ["ls", "read_file"],   # writer gets file access only
    # }

    # ── 2. CREATE THE TASK TOOL ──────────────────────────────────────────────
    # Factory returns (function, schema) — we add both to parent's registry.
    # BASE_TOOLS and BASE_SCHEMAS include all sub-agent tools too
    # so the factory can filter them when building each sub-agent.
    task_fn, task_schema = create_task_tool(
        sub_agent_configs=[research_sub_agent],
        all_tools=BASE_TOOLS,
        all_schemas=BASE_SCHEMAS,
    )

    # ── 3. ASSEMBLE PARENT TOOL REGISTRY ────────────────────────────────────
    # Parent gets:
    #   - All base tools (todos + files + think + direct search for trivial lookups)
    #   - The task tool (portal to all sub-agents)
    parent_tools = {
        **BASE_TOOLS,         # write_todos, read_todos, ls, write_file, read_file,
                              # think_tool, web_search
        "task": task_fn,      # the sub-agent delegation tool
    }

    parent_schemas = BASE_SCHEMAS + [task_schema]

    # ── 4. BUILD INITIAL STATE ───────────────────────────────────────────────
    # Just a plain dict — no TypedDict, no schema validation
    state = make_state()

    # Add the user's message to kick things off
    state["messages"].append({
        "role": "user",
        "content": (
            "Research what Python is and its main use cases. "
            "Give me a well-structured summary."
        ),
    })

    # ── 5. RUN THE PARENT AGENT ──────────────────────────────────────────────
    print_separator("Deep agent starting")
    print("Question: Research what Python is and its main use cases.")

    final_state = run_agent(
        state=state,
        tools=parent_tools,
        tool_schemas=parent_schemas,
        system_prompt=build_parent_prompt(max_parallel=3),
    )

    # ── 6. SHOW RESULTS ──────────────────────────────────────────────────────
    print_separator("Final answer")
    print_final_answer(final_state)
    print_state_summary(final_state)


if __name__ == "__main__":
    main()
