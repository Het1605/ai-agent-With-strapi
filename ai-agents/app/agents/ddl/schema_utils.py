"""
schema_utils.py — shared helpers for schema-related agents.
"""
from app.graph.state import AgentState


def format_history(state: AgentState, max_turns: int = 4) -> str:
    """
    Format the last `max_turns` conversation turns into a readable context block
    for injection into LLM prompts.

    Returns a string like:
        User: create collection
        Assistant: What should the collection be named?
        User: product
    or "(no prior conversation)" if history is empty.
    """
    history = (state.get("conversation_history") or [])[-max_turns:]
    if not history:
        return "(no prior conversation)"
    return "\n".join(
        f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
        for m in history
    )


def maybe_reset_schema(state: AgentState, new_table_name: str | None) -> bool:
    """
    Detects a table change and resets schema_data when the target collection
    switches mid-session, preventing cross-table column contamination.

    Call this AFTER the LLM has extracted the new table name but BEFORE
    merging any new columns into the current schema.

    Returns True if a reset was performed, False otherwise.
    """
    if not new_table_name:
        return False

    current_table = (state.get("schema_data") or {}).get("table_name")

    if current_table and current_table != new_table_name:
        print(f"\n[Schema Reset Triggered]")
        print(f"  Previous table : {current_table}")
        print(f"  New table      : {new_table_name}")
        print(f"  Clearing schema_data to avoid cross-table contamination.")
        state["schema_data"]      = {"table_name": new_table_name, "columns": []}
        state["missing_fields"]   = []
        state["interaction_phase"] = False
        return True

    return False
