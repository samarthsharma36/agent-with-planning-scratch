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


async def web_search(ctx: Context, query: str, max_results: int = 5) -> str:
    """Search the web for information on a topic.

    Be specific with your query — 'Python list comprehension syntax'
    is better than 'Python stuff'.

    Returns a summary. Full content is saved to files automatically.
    Use read_file() for full details.
    """
    from duckduckgo_search import DDGS

    results = DDGS().text(query, max_results=max_results)

    lines = []
    for r in results:
        lines.append(f"### {r['title']}\n{r['href']}\n{r['body']}")
    raw_content = "\n\n".join(lines) if lines else "No results found."

    safe_name = query.lower().replace(" ", "_")[:30]
    filename = f"search_{safe_name}.md"
    file_content = f"# Search result: {query}\n\n{raw_content}\n"

    async with ctx.store.edit_state() as s:
        if "files" not in s["state"]:
            s["state"]["files"] = {}
        s["state"]["files"][filename] = file_content

    summary = lines[0].split("\n")[0] if lines else "No results."
    return (
        f"Search complete for '{query}'. {len(results)} results.\n"
        f"Saved to: {filename}\n"
        f"Top result: {summary}\n"
        f"Use read_file('{filename}') for full details."
    )


# ═══════════════════════════════════════════════════════════════
# WRITER TOOLS (mock — stateless, no ctx needed)
# ═══════════════════════════════════════════════════════════════

def draft_section(topic: str) -> str:
    """Draft a written section on a given topic.

    Use when you need to produce structured prose for a report or document.
    Returns a complete section with intro, key points, and conclusion.
    """
    return f"[MOCK] Drafted section on '{topic}': Introduction... Key points... Conclusion."


def format_output(content: str) -> str:
    """Format raw content into clean markdown.

    Use after drafting to apply consistent structure, headings, and layout.
    """
    return f"[MOCK] Formatted output:\n## Result\n{content}\n---"


def check_grammar(text: str) -> str:
    """Check grammar and suggest improvements for a piece of text.

    Use before finalizing any written output to catch issues and
    assess readability.
    """
    return f"[MOCK] Grammar check for: '{text[:40]}...' — No issues found. Readability: High."


# ═══════════════════════════════════════════════════════════════
# ANALYST TOOLS (mock — stateless, no ctx needed)
# ═══════════════════════════════════════════════════════════════

def identify_patterns(data: str) -> str:
    """Identify recurring patterns in a body of text or data.

    Use when you need to find themes, trends, or repeated structures
    across a dataset or document.
    """
    return f"[MOCK] Patterns in '{data[:30]}...': Pattern A (frequent), Pattern B (occasional)."


def generate_insights(topic: str) -> str:
    """Generate key insights about a topic.

    Use to extract the core trend, notable outlier, and main implication
    from a topic or dataset.
    """
    return f"[MOCK] Insights on '{topic}': 1. Core trend. 2. Notable outlier. 3. Implication."


def score_relevance(item: str) -> str:
    """Score how relevant an item is to the current task.

    Use to prioritize which pieces of information or use cases are
    most important to include in the final output.
    """
    return f"[MOCK] Relevance score for '{item[:30]}': 8/10 — strongly relevant."


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

draft_section_tool     = FunctionTool.from_defaults(draft_section)
format_output_tool     = FunctionTool.from_defaults(format_output)
check_grammar_tool     = FunctionTool.from_defaults(check_grammar)
identify_patterns_tool = FunctionTool.from_defaults(identify_patterns)
generate_insights_tool = FunctionTool.from_defaults(generate_insights)
score_relevance_tool   = FunctionTool.from_defaults(score_relevance)
