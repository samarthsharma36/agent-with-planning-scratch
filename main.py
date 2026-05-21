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
from agent import create_researcher, create_writer, create_analyst, create_task_tool
from prompts import build_parent_prompt

load_dotenv()


async def main():
    # ── 1. SET UP THE LLM ────────────────────────────────────────────────────
    # Use Anthropic Claude via llama-index-llms-anthropic.
    # Setting Settings.tokenizer ensures accurate token counting throughout.
    llm = Anthropic(model="claude-sonnet-4-5-20250929")

    # ── 2. CREATE SUB-AGENTS ─────────────────────────────────────────────────
    researcher = create_researcher(llm)
    writer     = create_writer(llm)
    analyst    = create_analyst(llm)

    task_tool = create_task_tool({
        "research-agent": researcher,
        "writer-agent":   writer,
        "analyst-agent":  analyst,
    })

    # ── 3. CREATE THE PARENT AGENT ───────────────────────────────────────────
    parent = FunctionAgent(
        tools=[
            write_todos_tool,
            read_todos_tool,
            ls_tool,
            write_file_tool,
            read_file_tool,
            think_tool_tool,
            web_search_tool,
            task_tool,
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
            "Research Python's main use cases. "
            "Then have the writer format a clean summary report. "
            "Then have the analyst score the relevance of each use case."
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
