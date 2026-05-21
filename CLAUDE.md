# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency management with Python 3.13.

```bash
uv sync                        # install dependencies
uv run python main.py          # run the agent
source .venv/bin/activate && python main.py  # alternate run
```

Requires `ANTHROPIC_API_KEY` in a `.env` file (see `.env.example`).

## Architecture

An **educational LlamaIndex re-implementation** of a multi-agent system. Uses `llama_index.core.agent.workflow.FunctionAgent` — no manual ReAct loops, no raw Anthropic SDK.

**Reading order:** `tools.py` → `agent.py` → `main.py`

### Core components

**`tools.py`** — All tools as plain async functions. LlamaIndex auto-generates JSON schemas from type hints and docstrings — no hand-written schemas. Tools that need shared state take `ctx: Context` as the **first** parameter; LlamaIndex detects this and injects the agent's context automatically. Stateless tools (e.g. `think_tool`) omit `ctx` entirely.

State lives in `ctx.store` and is accessed via:
```python
async with ctx.store.edit_state() as s:
    s["state"]["todos"]  # task list
    s["state"]["files"]  # virtual filesystem {filename: content}
```

**`agent.py`** — Two factory functions:
- `create_researcher(llm)` — builds the research sub-agent (`FunctionAgent`) with only `web_search` + `think_tool`
- `create_task_tool(researcher)` — wraps `researcher.run()` as a `FunctionTool` the parent can call. Each call creates a fresh `Context` automatically (context isolation without any manual state management).

**`prompts.py`** — System prompts only. Tool descriptions are now docstrings on the functions in `tools.py`. `build_parent_prompt(max_parallel)` assembles three sections: TODO management, virtual filesystem, and delegation. `RESEARCHER_PROMPT` is the sub-agent's focused prompt.

**`main.py`** — Entry point: creates the LLM, builds researcher + task tool, assembles the parent `FunctionAgent` with all tools, runs it with streamed output via `stream_events()`.

### Key design patterns

- **Context injection**: Any tool function whose first parameter is `ctx: Context` automatically receives the live agent context — the LLM never passes this argument.
- **Context isolation**: Each `researcher.run()` call starts with a fresh `Context`. No manual `make_state()` / `merge_files()` needed.
- **Virtual filesystem**: `web_search` saves full results to `state["files"]` and returns only a summary; sub-agents can read files the parent wrote and vice versa.
- **`web_search` is mocked**: Returns canned results from `_MOCK_RESULTS`. Replace the function body with a Tavily/Serper call to make it real — docstring (schema) stays unchanged.

### Adding a new sub-agent

1. Add a system prompt in `prompts.py`
2. Create a `FunctionAgent` factory in `agent.py` with the allowed tools
3. Wrap its `.run()` with `FunctionTool.from_defaults()` and pass the tool to the parent in `main.py`

### Adding a new tool

1. Write `async def my_tool(ctx: Context, arg: str) -> str` in `tools.py` with a docstring describing when/how to use it
2. Add `my_tool_tool = FunctionTool.from_defaults(my_tool)` to the registry at the bottom
3. Import and add to the relevant agent's tool list in `agent.py` or `main.py`
