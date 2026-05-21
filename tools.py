"""
tools.py — All tools as plain async functions + FunctionTool wrappers

Each tool is just a Python function with type hints and a docstring.
LlamaIndex auto-generates the JSON schema from these — no manual schemas needed.

Tool signature rules:
  - ctx: Context  must be the FIRST parameter when the tool reads/writes state.
    LlamaIndex detects this and injects the current agent's context automatically.
  - All other parameters are what the LLM provides.
  - Return a plain string — this is what the LLM reads as the tool result.

State lives in ctx.store:
  - ctx_state["state"]["todos"]  — task list
  - ctx_state["state"]["files"]  — virtual filesystem {filename: content}
"""

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context


# ═══════════════════════════════════════════════════════════════
# TODO TOOLS
# ═══════════════════════════════════════════════════════════════

async def write_todos(ctx: Context, todos: list) -> str:
    """Manage your TODO list for multi-step tasks.

    WHEN to use:
    - At the very start: write your full plan before doing ANY work
    - After completing each step: update statuses and add newly discovered tasks
    - When your plan changes: rewrite the whole list

    FORMAT: todos is a list of dicts with 'content' (str) and 'status'
    ('pending', 'in_progress', or 'completed').

    CRITICAL: Always rewrite the FULL list. Never do partial updates.
    """
    async with ctx.store.edit_state() as s:
        s["state"]["todos"] = todos

    lines = []
    for i, todo in enumerate(todos, 1):
        emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(
            todo.get("status", "pending"), "❓"
        )
        lines.append(f"{i}. {emoji} {todo['content']} ({todo['status']})")

    return "TODO list updated:\n" + "\n".join(lines)


async def read_todos(ctx: Context) -> str:
    """Read the current TODO list.

    Use this to re-orient yourself when lost in a long task,
    or to check what's still pending before deciding next steps.
    """
    async with ctx.store.edit_state() as s:
        todos = s["state"].get("todos", [])

    if not todos:
        return "No todos in the list yet."

    lines = ["Current TODO list:"]
    for i, todo in enumerate(todos, 1):
        emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(
            todo.get("status", "pending"), "❓"
        )
        lines.append(f"{i}. {emoji} {todo['content']} ({todo['status']})")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# FILE TOOLS
# ═══════════════════════════════════════════════════════════════

async def ls(ctx: Context) -> str:
    """List all files in the virtual filesystem.

    Call this FIRST before starting any task to see what context already exists.
    Files persist across the entire conversation — previous work may be saved here.
    """
    async with ctx.store.edit_state() as s:
        files = list(s["state"].get("files", {}).keys())

    if not files:
        return "No files in the filesystem yet."

    return "Files:\n" + "\n".join(f"  - {f}" for f in files)


async def write_file(ctx: Context, file_path: str, content: str) -> str:
    """Write content to the virtual filesystem.

    WHEN to use:
    - After any search: save the full results immediately
    - When preserving intermediate work
    - Name files descriptively: 'python_overview.md', not 'file1.txt'

    Always save to files instead of keeping raw content in your context.
    """
    async with ctx.store.edit_state() as s:
        if "files" not in s["state"]:
            s["state"]["files"] = {}
        s["state"]["files"][file_path] = content

    return f"Saved: {file_path} ({len(content)} chars)"


async def read_file(
    ctx: Context,
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content from the virtual filesystem.

    WHEN to use:
    - When you need full details of a previously saved result
    - When writing a report (read research files first)
    - Use offset + limit for large files to read in chunks

    Args:
      file_path: exact filename from ls()
      offset: line number to start from (default 0)
      limit: max lines to read (default 2000)
    """
    async with ctx.store.edit_state() as s:
        files = s["state"].get("files", {})

    if file_path not in files:
        available = list(files.keys())
        return f"Error: '{file_path}' not found. Use ls() to see available files. Found: {available}"

    content = files[file_path]

    if not content:
        return f"File '{file_path}' exists but is empty."

    lines = content.splitlines()
    start = offset
    end   = min(start + limit, len(lines))

    if start >= len(lines):
        return f"Error: offset {offset} exceeds file length ({len(lines)} lines)"

    numbered = [f"{i+1:5d}\t{lines[i][:2000]}" for i in range(start, end)]
    header = f"File: {file_path} (lines {start+1}–{end} of {len(lines)} total)\n"
    return header + "\n".join(numbered)


# ═══════════════════════════════════════════════════════════════
# RESEARCH TOOLS
# ═══════════════════════════════════════════════════════════════

def think_tool(reflection: str) -> str:
    """Reflect on your progress before deciding the next step.

    Use after each search or major action:
    - What did I find? Is it sufficient?
    - What is still missing?
    - Should I search again or write my final answer now?

    Forces deliberate reasoning instead of jumping to conclusions.
    """
    return f"Reflection recorded: {reflection}"


# Mock web search — returns canned results so no external API key is needed.
# To make this real: replace the function body with a Tavily/Serper call.
# The docstring (schema) stays the same — the LLM doesn't care what's inside.
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
    """,
}


async def web_search(ctx: Context, query: str) -> str:
    """Search the web for information on a topic.

    Be specific with your query — 'Python list comprehension syntax'
    is better than 'Python stuff'.

    Returns a summary. Full content is saved to files automatically.
    Use read_file() for full details.
    """
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["use case", "application", "real world", "example"]):
        raw_content = _MOCK_RESULTS["use cases"]
    else:
        raw_content = _MOCK_RESULTS["default"]

    safe_name = query.lower().replace(" ", "_")[:30]
    filename = f"search_{safe_name}.md"

    file_content = f"# Search result: {query}\n\n## Content\n{raw_content.strip()}\n"

    async with ctx.store.edit_state() as s:
        if "files" not in s["state"]:
            s["state"]["files"] = {}
        s["state"]["files"][filename] = file_content

    summary = raw_content.strip().split("\n")[0][:200]
    return (
        f"Search complete for '{query}'.\n"
        f"Saved full results to: {filename}\n"
        f"Summary: {summary}\n"
        f"Use read_file('{filename}') for full details."
    )


# ═══════════════════════════════════════════════════════════════
# TOOL REGISTRY — what agent.py and main.py import
# ═══════════════════════════════════════════════════════════════

write_todos_tool = FunctionTool.from_defaults(write_todos)
read_todos_tool  = FunctionTool.from_defaults(read_todos)
ls_tool          = FunctionTool.from_defaults(ls)
write_file_tool  = FunctionTool.from_defaults(write_file)
read_file_tool   = FunctionTool.from_defaults(read_file)
think_tool_tool  = FunctionTool.from_defaults(think_tool)
web_search_tool  = FunctionTool.from_defaults(web_search)
