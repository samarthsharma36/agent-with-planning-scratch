"""
loop.py — The ReAct loop engine

This is the single most important file in the project.
This is EXACTLY what LangGraph's create_agent() was doing internally.

The loop:
  1. Call the LLM with current messages + tools
  2. LLM returns a response
  3. If stop_reason == "end_turn"  → LLM is done, no tool calls → exit loop
  4. If stop_reason == "tool_use"  → LLM wants to use tools → run them
  5. Append tool results to messages as a "user" turn
  6. Go back to step 1

That's it. That's the entire ReAct loop.
Every AI agent framework is just a wrapper around this pattern.
"""

import anthropic
from dotenv import load_dotenv

load_dotenv()

# One client, shared across all agents (parent + all sub-agents)
client = anthropic.Anthropic()

# Model used by all agents
# Change this to switch every agent at once
MODEL = "claude-sonnet-4-5-20250929"

# Safety limit — prevents infinite loops if something goes wrong
MAX_TURNS = 20


def run_agent(
    state: dict,
    tools: dict,
    tool_schemas: list,
    system_prompt: str,
    max_turns: int = MAX_TURNS,
) -> dict:
    """
    Run the ReAct loop until the LLM stops calling tools.

    LangGraph equivalent:
        agent = create_agent(model, tools, system_prompt, state_schema)
        result = agent.invoke(initial_state)

    Here we do it ourselves — explicit loop, no magic.

    Args:
        state:        the agent's state dict (mutated in place AND returned)
        tools:        dict mapping tool_name → function
                      e.g. {"write_todos": write_todos_fn, "ls": ls_fn}
        tool_schemas: list of JSON schema dicts for the API
                      e.g. [WRITE_TODOS_SCHEMA, LS_SCHEMA, ...]
        system_prompt: the agent's system prompt string
        max_turns:    safety limit on loop iterations

    Returns:
        The final state dict (same object as input, mutated)
    """

    for turn in range(max_turns):

        # ── STEP 1: Call the LLM ────────────────────────────────────────────
        # Pass the full conversation history every time.
        # The LLM has no memory — we replay the entire context on each call.
        # This is true whether you use LangGraph or raw SDK.
        print(f"  [turn {turn + 1}] calling LLM...", flush=True)
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system_prompt,
            tools=tool_schemas if tool_schemas else [],
            messages=state["messages"],
        )

        # ── STEP 2: Store the assistant's response ──────────────────────────
        # Convert ContentBlock objects → plain dicts for consistent storage.
        # We'll append these to state["messages"] as the assistant turn.
        content_dicts = _blocks_to_dicts(response.content)
        state["messages"].append({
            "role": "assistant",
            "content": content_dicts,
        })

        # ── STEP 3: Check stop reason ───────────────────────────────────────
        if response.stop_reason == "end_turn":
            # LLM produced a final text response with no tool calls.
            # Loop ends here. State is complete.
            break

        if response.stop_reason == "tool_use":
            # LLM wants to call one or more tools.
            # Process ALL tool calls from this response, then loop back.

            tool_results = []  # collect all results before appending

            for block in response.content:
                if block.type != "tool_use":
                    continue  # skip text blocks in this pass

                tool_name    = block.name    # e.g. "write_todos"
                tool_input   = block.input   # dict of LLM-provided args
                tool_call_id = block.id      # unique ID for this call
                                             # e.g. "toolu_01XYZ..."
                print(f"  [tool] {tool_name}", flush=True)

                # ── STEP 4: Execute the tool ─────────────────────────────────
                # This is what InjectedState and InjectedToolCallId did.
                # LangGraph injected these automatically.
                # Here we pass them explicitly — that's all it was doing.
                result = _execute_tool(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_call_id=tool_call_id,
                    state=state,
                    tools=tools,
                )

                # Each result links back to its tool call via tool_use_id
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,  # must match block.id above
                    "content": str(result),
                })

            # ── STEP 5: Append all results as a user turn ───────────────────
            # The Anthropic API requires tool results in a "user" role message.
            # This is how the LLM "sees" what the tools returned.
            state["messages"].append({
                "role": "user",
                "content": tool_results,
            })

            # Loop back to step 1 — LLM reads results and decides next action

    return state


def _execute_tool(
    tool_name: str,
    tool_input: dict,
    tool_call_id: str,
    state: dict,
    tools: dict,
) -> str:
    """
    Look up and call a tool by name, injecting state and tool_call_id.

    This is the demystification of InjectedState and InjectedToolCallId.
    LangGraph detected those annotations and injected the values automatically.
    Here we just... pass them as keyword arguments. That's all it was.

    Args:
        tool_name:    name of the tool to call
        tool_input:   dict of arguments the LLM provided
        tool_call_id: the tool call's unique ID (for linking responses)
        state:        current agent state (we inject this into every tool)
        tools:        registry of available tools

    Returns:
        String result to send back to the LLM
    """
    if tool_name not in tools:
        # Return a helpful error — the LLM can read this and recover
        return f"Error: tool '{tool_name}' not found. Available: {list(tools.keys())}"

    try:
        # Call the tool function
        # Every tool signature is: fn(state, tool_call_id, **llm_provided_args)
        # state and tool_call_id are OUR additions — LLM never sees them in schema
        result = tools[tool_name](
            state=state,                # InjectedState equivalent
            tool_call_id=tool_call_id,  # InjectedToolCallId equivalent
            **tool_input,               # everything the LLM decided to pass
        )
        return result if result is not None else ""

    except Exception as e:
        # Return error as string — LLM can read this and retry
        return f"Error running '{tool_name}': {str(e)}"


def _blocks_to_dicts(content_blocks) -> list:
    """
    Convert Anthropic SDK ContentBlock objects → plain Python dicts.

    The SDK returns typed objects (TextBlock, ToolUseBlock).
    We convert to dicts for consistent, serializable storage in state.

    Args:
        content_blocks: list of ContentBlock objects from response.content

    Returns:
        list of plain dicts with type, text, id, name, input fields
    """
    result = []
    for block in content_blocks:
        if block.type == "text":
            result.append({
                "type": "text",
                "text": block.text,
            })
        elif block.type == "tool_use":
            result.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return result
