"""
state.py — Replaces DeepAgentState + all LangGraph reducers

In LangGraph you defined a TypedDict with Annotated fields and reducer functions.
LangGraph would call those reducers automatically when state was updated.

Here we just use a plain Python dict.
WE call the merge logic ourselves — no magic, no framework.

State shape:
{
    "messages": [...],   # full conversation history
    "todos":    [...],   # current task list
    "files":    {...},   # virtual filesystem  {filename: content}
}
"""


def make_state() -> dict:
    """
    Create a fresh empty state dict.

    LangGraph equivalent:
        state = DeepAgentState(messages=[], todos=[], files={})

    Here it's just a dict. Nothing special.
    """
    return {
        "messages": [],  # conversation history — list of {role, content} dicts
        "todos":    [],  # task list — list of {content, status} dicts
        "files":    {},  # virtual filesystem — {filename: file_content_string}
    }


def merge_files(state: dict, new_files: dict) -> None:
    """
    Merge new files into state, with new values winning on conflicts.

    LangGraph equivalent:
        def file_reducer(left, right):
            return {**left, **right}
        files: Annotated[dict, file_reducer]

    LangGraph called this reducer automatically when a Command updated files.
    Here we call it ourselves — explicit is better than magic.

    Args:
        state:     the current state dict (mutated in place)
        new_files: new files to merge in — these overwrite existing keys
    """
    state["files"].update(new_files)  # right side wins, same as {**left, **right}


def append_message(state: dict, role: str, content) -> None:
    """
    Append a message to the conversation history.

    LangGraph equivalent:
        messages: Annotated[list, add_messages]

    add_messages was a reducer that appended new messages to the list.
    Here we just call list.append() ourselves.

    Args:
        state:   the current state dict (mutated in place)
        role:    "user" or "assistant"
        content: string or list of content blocks
    """
    state["messages"].append({"role": role, "content": content})


def get_last_assistant_text(state: dict) -> str:
    """
    Extract the final text response from the last assistant message.

    Used by the task tool to pull the sub-agent's final answer
    out of its completed state and return it to the parent.

    Args:
        state: a completed agent state

    Returns:
        The last text string the assistant produced, or empty string
    """
    # Walk messages in reverse — find the last assistant message
    for msg in reversed(state["messages"]):
        if msg["role"] == "assistant":
            content = msg["content"]

            # Content can be a list of blocks (normal case)
            if isinstance(content, list):
                # Walk blocks in reverse — find the last text block
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")

            # Content can be a plain string (edge case)
            elif isinstance(content, str):
                return content

    return ""  # no assistant message found
