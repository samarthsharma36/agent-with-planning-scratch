"""
agent.py — Sub-agent factory

Replaces _create_task_tool() from the LangGraph version.

Same pattern: a factory function that builds and returns the task tool.
The factory captures the sub-agent registry in a closure —
so the task function knows which agents are available without globals.

The key difference from LangGraph:
  - No create_react_agent — we use our own run_agent() loop
  - No InjectedState / Command — we pass state directly and mutate it
  - Sub-agents are just run_agent() calls with isolated message history
"""

from state import make_state, get_last_assistant_text, merge_files
from loop import run_agent
from prompts import TASK_DESCRIPTION_TEMPLATE


def create_task_tool(
    sub_agent_configs: list,
    all_tools: dict,
    all_schemas: list,
) -> tuple:
    """
    Factory that builds the task() tool for sub-agent delegation.

    LangGraph equivalent:
        task_tool = _create_task_tool(tools, subagents, model, state_schema)

    Returns a (function, schema) tuple instead of a single tool object,
    because we need both pieces separately in our system.

    Args:
        sub_agent_configs: list of sub-agent config dicts, each with:
                           - name:        str (used in task() calls)
                           - description: str (shown to parent in tool desc)
                           - prompt:      str (sub-agent's system prompt)
                           - tools:       list[str] (tool names it can use)
        all_tools:   dict of ALL available tool functions (name → fn)
        all_schemas: list of ALL available tool schemas

    Returns:
        (task_fn, task_schema) — function and its JSON schema
    """

    # ── Build the agent registry ─────────────────────────────────────────────
    # Same as the LangGraph version — a dict of name → config
    # In LangGraph this was: name → actual LangGraph agent object
    # Here it's: name → config dict (we run run_agent() lazily on each call)
    agents = {config["name"]: config for config in sub_agent_configs}

    # ── Build the description string for the task tool ───────────────────────
    # This is what the parent LLM reads to know which agents exist
    agents_list = "\n".join(
        f"  - {c['name']}: {c['description']}"
        for c in sub_agent_configs
    )

    # ── Build the task function (closure over agents registry) ───────────────
    def task_fn(state: dict, tool_call_id: str, description: str, subagent_type: str) -> str:
        """
        Delegate a task to a specialist sub-agent.

        LangGraph equivalent of the inner task() function in _create_task_tool.

        Steps:
          1. Validate the requested agent type
          2. Create isolated state (fresh messages, carry over files)
          3. Filter tools to only what this sub-agent needs
          4. Run the sub-agent loop (run_agent)
          5. Merge sub-agent's files back into parent state
          6. Return sub-agent's final text as a string
        """

        # ── Step 1: Validate ─────────────────────────────────────────────────
        if subagent_type not in agents:
            return (
                f"Error: '{subagent_type}' not found. "
                f"Available agents: {list(agents.keys())}"
            )

        config = agents[subagent_type]

        # ── Step 2: Context isolation ─────────────────────────────────────────
        # This is THE most important step. This is what makes sub-agents work.
        #
        # LangGraph equivalent:
        #     state["messages"] = [{"role": "user", "content": description}]
        #
        # We create a FRESH state dict with:
        #   - messages: only the task description (nothing from parent history)
        #   - todos:    empty (sub-agent starts with no todos)
        #   - files:    COPIED from parent (sub-agent can read parent's files)
        #
        # Why copy files but not messages?
        #   - Files are the "long term memory" (intentionally shared)
        #   - Messages are the "working memory" (intentionally isolated)
        sub_state = make_state()
        sub_state["messages"] = [{"role": "user", "content": description}]
        sub_state["files"]    = dict(state.get("files", {}))  # shallow copy

        # ── Step 3: Filter tools for this sub-agent ──────────────────────────
        # Each sub-agent gets only the tools it needs.
        # A researcher gets search tools. A writer gets file tools.
        # This keeps each agent focused and prevents misuse.
        allowed_tool_names = config.get("tools", list(all_tools.keys()))

        sub_tools = {
            name: fn
            for name, fn in all_tools.items()
            if name in allowed_tool_names
        }

        sub_schemas = [
            schema
            for schema in all_schemas
            if schema["name"] in allowed_tool_names
        ]

        # ── Step 4: Run the sub-agent ─────────────────────────────────────────
        # This is JUST our run_agent() loop — same one the parent uses.
        # LangGraph was doing exactly this: running the same loop in isolation.
        result_state = run_agent(
            state=sub_state,
            tools=sub_tools,
            tool_schemas=sub_schemas,
            system_prompt=config["prompt"],
        )

        # ── Step 5: Merge files back to parent ────────────────────────────────
        # Files the sub-agent created (research results, notes, etc.)
        # need to be visible to the parent for reading later.
        #
        # LangGraph equivalent:
        #     return Command(update={"files": result.get("files", {})})
        #
        # Here we call merge_files() directly — same logic, no Command needed.
        merge_files(state, result_state.get("files", {}))

        # ── Step 6: Extract and return final text ─────────────────────────────
        # The sub-agent's final answer becomes a ToolMessage in the parent's context.
        # LangGraph: ToolMessage(result["messages"][-1].content, tool_call_id=...)
        # Here: we return a string, the loop wraps it in tool_result automatically.
        last_text = get_last_assistant_text(result_state)
        return last_text if last_text else "Sub-agent completed with no text response."

    # ── Build the JSON schema for the task tool ──────────────────────────────
    # The schema is built dynamically — agent names go into an enum
    # so the LLM can only pick valid agent types (no typos)
    task_schema = {
        "name": "task",
        "description": TASK_DESCRIPTION_TEMPLATE.format(agents_list=agents_list),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Complete self-contained task description. "
                        "Sub-agent has NO memory of this conversation — "
                        "include ALL context it needs."
                    ),
                },
                "subagent_type": {
                    "type": "string",
                    "enum": list(agents.keys()),   # only valid agent names allowed
                    "description": "Which specialist to delegate to",
                },
            },
            "required": ["description", "subagent_type"],
        },
    }

    return task_fn, task_schema
