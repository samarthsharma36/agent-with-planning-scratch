"""
main.py — Wire everything together and run the deep agent

Reading order for understanding:
  tools.py  →  agent.py  →  main.py (here)

Everything flows into main.py. Nothing flows out.
"""

import asyncio
from dotenv import load_dotenv
from llama_index.llms.anthropic import Anthropic
from llama_index.core.agent.workflow import FunctionAgent, AgentOutput, ToolCall

from tools import (
    write_todos_tool,
    read_todos_tool,
    ls_tool,
    write_file_tool,
    read_file_tool,
    think_tool_tool,
    web_search_tool,
)
from agent import create_researcher, create_task_tool
from prompts import build_parent_prompt

load_dotenv()


async def main():
    # ── 1. SET UP THE LLM ────────────────────────────────────────────────────
    # Use Anthropic Claude via llama-index-llms-anthropic.
    # Setting Settings.tokenizer ensures accurate token counting throughout.
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    # ── 2. CREATE THE RESEARCH SUB-AGENT ─────────────────────────────────────
    # The researcher is a FunctionAgent with web_search + think_tool.
    # create_task_tool() wraps researcher.run() as a FunctionTool so the
    # parent agent can delegate to it by calling task(description=...).
    researcher = create_researcher(llm)
    task_tool  = create_task_tool(researcher)

    # ── 3. CREATE THE PARENT AGENT ───────────────────────────────────────────
    # Parent gets all base tools + the task delegation tool.
    parent = FunctionAgent(
        tools=[
            write_todos_tool,   # plan with todos
            read_todos_tool,
            ls_tool,            # virtual filesystem
            write_file_tool,
            read_file_tool,
            think_tool_tool,    # structured reflection
            web_search_tool,    # can search directly for trivial lookups
            task_tool,          # delegate to research-agent
        ],
        llm=llm,
        system_prompt=build_parent_prompt(max_parallel=3),
    )

    # ── 4. RUN ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Deep agent starting")
    print("=" * 60 + "\n")

    handler = parent.run(
        user_msg=(
            "Research what Python is and its main use cases. "
            "Give me a well-structured summary."
        )
    )

    # Stream events so the user can see progress instead of a silent hang.
    # ToolCall fires each time the agent invokes a tool.
    # AgentStream fires as the final text response is generated token by token.
    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            print(f"  [tool] {event.tool_name}", flush=True)
        elif isinstance(event, AgentOutput):
            print(event.response, flush=True)

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
