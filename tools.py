"""
tools.py — All tools as plain functions + their JSON schemas

In LangGraph you used @tool decorator + InjectedState + InjectedToolCallId + Command.
Here each tool is just TWO things:

  1. A plain Python function:
       - signature: fn(state, tool_call_id, ...llm_args...)
       - state and tool_call_id are injected by the loop (not from LLM)
       - mutates state directly instead of returning a Command
       - returns a plain string (the message that goes back to the LLM)

  2. A JSON schema dict:
       - tells the Anthropic API what the tool does and what args it takes
       - only includes LLM-provided args (NOT state or tool_call_id)
       - this is what the LLM reads to decide how to call the tool

The @tool decorator in LangChain was doing two things:
  - generating the JSON schema from the function signature + docstring
  - wrapping the function for LangGraph's tool node
Here we write the schema manually — more work but zero magic.
"""

from prompts import (
    WRITE_TODOS_DESCRIPTION,
    READ_TODOS_DESCRIPTION,
    LS_DESCRIPTION,
    READ_FILE_DESCRIPTION,
    WRITE_FILE_DESCRIPTION,
    THINK_DESCRIPTION,
    WEB_SEARCH_DESCRIPTION,
)


# ═══════════════════════════════════════════════════════════════
# PILLAR 1 — TODO TOOLS
# ═══════════════════════════════════════════════════════════════

def write_todos(state: dict, tool_call_id: str, todos: list) -> str:
    """
    Overwrite the entire TODO list in state.

    LangGraph equivalent:
        return Command(update={"todos": todos, "messages": [ToolMessage(...)]})

    Here we:
      - directly mutate state["todos"]  (no Command needed)
      - return a string                 (loop puts this in a tool_result message)

    The string we return IS the ToolMessage content.
    The loop wraps it in {"type": "tool_result", "tool_use_id": ..., "content": ...}
    automatically.
    """
    # Directly update state — no Command, no reducer, just assignment
    state["todos"] = todos

    # Format for readable confirmation
    lines = []
    for i, todo in enumerate(todos, 1):
        emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(
            todo.get("status", "pending"), "❓"
        )
        lines.append(f"{i}. {emoji} {todo['content']} ({todo['status']})")

    return "TODO list updated:\n" + "\n".join(lines)


WRITE_TODOS_SCHEMA = {
    "name": "write_todos",
    "description": WRITE_TODOS_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Complete TODO list (replaces existing list entirely)",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Short specific task description",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Current task status",
                        },
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    },
}


def read_todos(state: dict, tool_call_id: str) -> str:
    """
    Read the current TODO list from state.

    LangGraph equivalent:
        todos = state.get("todos", [])
        return formatted string

    No state changes — just reads and returns.
    No Command needed since we're not updating anything.
    """
    todos = state.get("todos", [])

    if not todos:
        return "No todos in the list yet."

    lines = ["Current TODO list:"]
    for i, todo in enumerate(todos, 1):
        emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(
            todo.get("status", "pending"), "❓"
        )
        lines.append(f"{i}. {emoji} {todo['content']} ({todo['status']})")

    return "\n".join(lines)


READ_TODOS_SCHEMA = {
    "name": "read_todos",
    "description": READ_TODOS_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {},   # no LLM-provided args — reads from injected state
        "required": [],
    },
}


# ═══════════════════════════════════════════════════════════════
# PILLAR 2 — FILE TOOLS
# ═══════════════════════════════════════════════════════════════

def ls(state: dict, tool_call_id: str) -> str:
    """
    List all files in the virtual filesystem.

    LangGraph equivalent:
        return list(state.get("files", {}).keys())

    Simple read — no state changes, returns a formatted string.
    """
    files = list(state.get("files", {}).keys())

    if not files:
        return "No files in the filesystem yet."

    return "Files:\n" + "\n".join(f"  - {f}" for f in files)


LS_SCHEMA = {
    "name": "ls",
    "description": LS_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {},   # no args — reads from injected state
        "required": [],
    },
}


def write_file(state: dict, tool_call_id: str, file_path: str, content: str) -> str:
    """
    Write content to the virtual filesystem.

    LangGraph equivalent:
        files = state.get("files", {})
        files[file_path] = content
        return Command(update={"files": files, "messages": [ToolMessage(...)]})

    Here we:
      - directly mutate state["files"]  (the file_reducer merge is just dict assignment)
      - return a string confirmation

    Note: state["files"] is already a dict. We set a key directly.
    The merge_files() helper in state.py exists for when we need to merge
    a whole dict (like when sub-agent files come back). Here it's one file.
    """
    # Direct mutation — replaces Command + file_reducer
    state["files"][file_path] = content

    return f"Saved: {file_path} ({len(content)} chars)"


WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": WRITE_FILE_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Descriptive filename e.g. 'python_overview.md'",
            },
            "content": {
                "type": "string",
                "description": "Full content to save",
            },
        },
        "required": ["file_path", "content"],
    },
}


def read_file(
    state: dict,
    tool_call_id: str,
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """
    Read file content from the virtual filesystem with optional pagination.

    LangGraph equivalent:
        files = state.get("files", {})
        if file_path not in files:
            return f"Error: ..."
        lines = files[file_path].splitlines()
        ...

    Exactly the same logic — just no @tool decorator around it.
    """
    files = state.get("files", {})

    if file_path not in files:
        # Error message is FOR THE LLM — it can use this to retry with correct name
        available = list(files.keys())
        return f"Error: '{file_path}' not found. Use ls() to see available files. Found: {available}"

    content = files[file_path]

    if not content:
        return f"File '{file_path}' exists but is empty."

    # Split into lines and apply pagination
    lines = content.splitlines()
    start = offset
    end   = min(start + limit, len(lines))

    if start >= len(lines):
        return f"Error: offset {offset} exceeds file length ({len(lines)} lines)"

    # Format with line numbers so LLM can reference specific lines
    numbered = [f"{i+1:5d}\t{lines[i][:2000]}" for i in range(start, end)]

    header = f"File: {file_path} (lines {start+1}–{end} of {len(lines)} total)\n"
    return header + "\n".join(numbered)


READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": READ_FILE_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Exact filename from ls()",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start from (default: 0)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to read (default: 2000)",
            },
        },
        "required": ["file_path"],
    },
}


# ═══════════════════════════════════════════════════════════════
# RESEARCH TOOLS
# ═══════════════════════════════════════════════════════════════

def think_tool(state: dict, tool_call_id: str, reflection: str) -> str:
    """
    Structured reflection — forces deliberate reasoning before acting.

    LangGraph equivalent: identical, just without @tool decorator.

    No state changes — just echoes the reflection back so the LLM
    knows its thinking was recorded and can continue reasoning.
    """
    # We don't store this anywhere — just echo it back.
    # The act of writing the reflection IS the value.
    # It forces the LLM to articulate its reasoning before the next action.
    return f"Reflection recorded: {reflection}"


THINK_SCHEMA = {
    "name": "think_tool",
    "description": THINK_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "reflection": {
                "type": "string",
                "description": "Your detailed analysis of findings and next steps",
            }
        },
        "required": ["reflection"],
    },
}


# Mock web search — returns canned results so no Tavily API key needed.
# In production: replace this function body with a real Tavily/Serper call.
# The schema stays identical — the LLM doesn't know or care what's inside.
_MOCK_RESULTS = {
    "default": """
Python is a high-level, interpreted programming language created by Guido van Rossum,
first released in 1991. It emphasizes code readability and simplicity.

Key features:
- Dynamic typing and automatic memory management
- Extensive standard library ("batteries included")
- Supports multiple programming paradigms: procedural, OOP, functional
- Large ecosystem: NumPy, Pandas, Django, Flask, PyTorch, TensorFlow

Common use cases:
- Web development (Django, Flask, FastAPI)
- Data science and machine learning (NumPy, Pandas, scikit-learn)
- Automation and scripting
- APIs and backend services

Python 3 is the current version. Python 2 reached end-of-life in 2020.
The language is consistently ranked among the top 3 most popular languages worldwide.
""",
    "use cases": """
Python real-world use cases and applications:

1. Data Science: Used by Netflix for recommendation engines, by Spotify for music analysis
2. Machine Learning: PyTorch (Meta), TensorFlow (Google) are Python-first frameworks
3. Web Development: Instagram runs on Django, Dropbox uses Python extensively
4. Automation: DevOps pipelines, test automation, data pipelines
5. Scientific Computing: NASA, CERN use Python for data analysis
6. Finance: Quant trading algorithms, risk modeling at major banks
7. Education: Most universities teach Python as first language

Python's "batteries included" philosophy means most tasks need no external libraries.
""",
}


def web_search(state: dict, tool_call_id: str, query: str) -> str:
    """
    Search the web (mock version — returns canned data).

    LangGraph equivalent: the tavily_search tool with context offloading.

    Key pattern: save FULL content to a file, return ONLY a summary.
    This is context offloading — keeps the LLM's active context lean.

    To make this real: replace the mock result with an actual API call.
    Everything else stays exactly the same.
    """
    # Pick a mock result based on query keywords
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["use case", "application", "real world", "example"]):
        raw_content = _MOCK_RESULTS["use cases"]
    else:
        raw_content = _MOCK_RESULTS["default"]

    # Generate a filename from the query
    # In production: the summarization model generates this
    safe_name = query.lower().replace(" ", "_")[:30]
    filename = f"search_{safe_name}.md"

    # Build the full file content
    file_content = f"""# Search result: {query}

## Content
{raw_content.strip()}
"""

    # Save full content to the virtual filesystem (context offloading)
    # LangGraph version returned Command(update={"files": ..., "messages": [...]})
    # Here we directly mutate state["files"] and return a summary string
    state["files"][filename] = file_content

    # Return ONLY a minimal summary — not the full content
    # This is the core of context offloading:
    # The LLM gets a 2-line summary, not 500 words of raw content
    summary = raw_content.strip().split("\n")[0][:200]
    return (
        f"Search complete for '{query}'.\n"
        f"Saved full results to: {filename}\n"
        f"Summary: {summary}\n"
        f"Use read_file('{filename}') for full details."
    )


WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": WEB_SEARCH_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Specific search query",
            }
        },
        "required": ["query"],
    },
}


# ═══════════════════════════════════════════════════════════════
# REGISTRIES — what the rest of the system imports
# ═══════════════════════════════════════════════════════════════

# All base tools as a dict: name → function
# This is what the loop uses to look up and call tools by name
BASE_TOOLS = {
    "write_todos": write_todos,
    "read_todos":  read_todos,
    "ls":          ls,
    "write_file":  write_file,
    "read_file":   read_file,
    "think_tool":  think_tool,
    "web_search":  web_search,
}

# All base schemas as a list
# This is what gets passed to client.messages.create(tools=...)
BASE_SCHEMAS = [
    WRITE_TODOS_SCHEMA,
    READ_TODOS_SCHEMA,
    LS_SCHEMA,
    WRITE_FILE_SCHEMA,
    READ_FILE_SCHEMA,
    THINK_SCHEMA,
    WEB_SEARCH_SCHEMA,
]
