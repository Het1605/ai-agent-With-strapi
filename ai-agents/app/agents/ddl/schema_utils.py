"""
schema_utils.py — shared helpers for schema-related agents.
"""
from app.graph.state import AgentState


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
