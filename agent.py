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

from tools import (
    web_search_tool, think_tool_tool,
    draft_section_tool, format_output_tool, check_grammar_tool,
    identify_patterns_tool, generate_insights_tool, score_relevance_tool,
)
from prompts import RESEARCHER_PROMPT, WRITER_PROMPT, ANALYST_PROMPT


def create_researcher(llm) -> FunctionAgent:
    """Build the research specialist agent."""
    return FunctionAgent(
        name="research-agent",
        description="Searches the web for a specific topic. Give one topic at a time.",
        tools=[web_search_tool, think_tool_tool],
        llm=llm,
        system_prompt=RESEARCHER_PROMPT,
    )


def create_writer(llm) -> FunctionAgent:
    """Build the writer specialist agent."""
    return FunctionAgent(
        name="writer-agent",
        description="Drafts, formats, and checks written text.",
        tools=[draft_section_tool, format_output_tool, check_grammar_tool],
        llm=llm,
        system_prompt=WRITER_PROMPT,
    )


def create_analyst(llm) -> FunctionAgent:
    """Build the analyst specialist agent."""
    return FunctionAgent(
        name="analyst-agent",
        description="Identifies patterns, generates insights, and scores relevance.",
        tools=[identify_patterns_tool, generate_insights_tool, score_relevance_tool],
        llm=llm,
        system_prompt=ANALYST_PROMPT,
    )


def create_task_tool(agents: dict) -> FunctionTool:
    """Wrap multiple specialist agents as a single routing tool.

    The parent calls task(agent=..., description=...) to delegate to any
    registered specialist. Each call creates a fresh Context automatically.
    """

    async def task(agent: str, description: str) -> str:
        """Delegate a task to a named specialist agent.

        Available specialists:
          - research-agent: Searches the web for a specific topic.
          - writer-agent:   Drafts, formats, and checks written text.
          - analyst-agent:  Identifies patterns, generates insights, scores relevance.

        HOW to use:
        - agent: exact specialist name from the list above.
        - description: Write the FULL context the agent needs.
          It has NO memory of this conversation — include ALL context.
        - One focused job per call (not 'do everything').
        - Call task() multiple times in one step for parallel jobs.
        """
        if agent not in agents:
            return f"Error: unknown agent '{agent}'. Choose from: {list(agents.keys())}"
        result = await agents[agent].run(user_msg=description)
        return str(result)

    return FunctionTool.from_defaults(task)
