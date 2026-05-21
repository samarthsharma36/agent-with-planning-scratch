"""
agent.py — Sub-agent factory

Creates the research specialist as a FunctionAgent and exposes its
run() method as a FunctionTool the parent agent can call.

The key difference from the old Anthropic SDK version:
  - No manual run_agent() loop — FunctionAgent handles the loop internally
  - No make_state() / merge_files() — each researcher.run() call starts
    with a fresh Context automatically (context isolation for free)
  - No tool filtering boilerplate — just pass the allowed tools directly

Reading order:
  tools.py → agent.py (here) → main.py
"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

from tools import web_search_tool, think_tool_tool
from prompts import RESEARCHER_PROMPT


def create_researcher(llm) -> FunctionAgent:
    """Build the research specialist agent.

    The researcher gets only the tools it needs:
      - web_search: to fetch information
      - think_tool: to reason about what it found before responding

    LlamaIndex equivalent of the old sub_agent_config dict +
    the inner run_agent() call in task_fn.
    """
    return FunctionAgent(
        name="research-agent",
        description="Searches the web for a specific topic. Give one topic at a time.",
        tools=[web_search_tool, think_tool_tool],
        llm=llm,
        system_prompt=RESEARCHER_PROMPT,
    )


def create_task_tool(researcher: FunctionAgent) -> FunctionTool:
    """Wrap the researcher's run() as a tool the parent agent can call.

    LlamaIndex equivalent of create_task_tool() in the old agent.py.

    The parent calls task(description=...) and gets back a string result.
    Each call creates a fresh Context for the researcher automatically —
    no manual state isolation needed.
    """

    async def task(description: str) -> str:
        """Delegate a focused research task to the specialist research agent.

        Available specialist:
          - research-agent: Searches the web for a specific topic.

        HOW to use:
        - description: Write the FULL context the agent needs.
          It has NO memory of this conversation — include ALL context.
        - One focused topic per call (not 'do everything').
        - Call task() multiple times in one step for parallel topics.
        """
        result = await researcher.run(user_msg=description)
        return str(result)

    return FunctionTool.from_defaults(task)
