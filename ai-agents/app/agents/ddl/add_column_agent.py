from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from app.agents.ddl.schema_utils import maybe_reset_schema
import json
import re


def _resolve_relation_uid(raw_target: str) -> str:
    noise = r'\b(table|collection|model|entity|db|database|the|a|an)\b'
    cleaned = re.sub(noise, '', raw_target, flags=re.IGNORECASE).strip()
    singular = re.sub(r'[\s\-_]+', '_', cleaned.lower()).strip('_')
    if singular.endswith('sses') or singular.endswith('ches') or singular.endswith('xes'):
        pass
    elif singular.endswith('ies'):
        singular = singular[:-3] + 'y'
    elif singular.endswith('s') and not singular.endswith('ss'):
        singular = singular[:-1]
    return f"api::{singular}.{singular}"


async def add_column_agent(state: AgentState) -> AgentState:
    """
    AddColumnAgent: extracts NEW fields/columns to add to an existing collection.
    Routes → interaction_planner (missing info) or → query_builder (complete).
    """
    print("\n----- ENTERING AddColumnAgent -----")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    user_input     = state.get("user_input", "")
    field_registry = state.get("field_registry", {})
    raw_schema     = state.get("schema_data") or {}
    current_schema = {
        "table_name": raw_schema.get("table_name") or None,
        "columns":    list(raw_schema.get("columns") or [])
    }

    print(f"[AddColumnAgent] user_input : {user_input}")

    system_prompt = (
        "You are a Database Field Extraction Specialist.\n\n"
        f"Strapi Field Registry: {json.dumps(field_registry)}\n\n"
        "Instructions:\n"
        "- Extract the TARGET collection name (entity noun: 'product', 'order', etc.).\n"
        "- NEVER use generic words as collection names: 'collection', 'table', 'add', 'column', 'field'.\n"
        "- If no clear collection name is present, set table_name to null.\n"
        "- Extract the NEW fields to be added.\n"
        "- Infer types: name/title→string, description/bio→text, email→email, "
        "password→password, price/amount/cost→decimal, age/count/stock/quantity→integer, "
        "date/created_at→datetime, is_*/active/verified→boolean.\n"
        "- For ENUM: extract enum values.\n"
        "- For RELATION: extract relation type and target.\n"
        "- NEVER add constraints unless user explicitly states them.\n\n"
        'Output ONLY valid JSON: {"extracted_data": {"table_name": <str|null>, "columns": [...]}}'
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User request: {user_input}")
    ])

    try:
        clean     = response.content.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(clean).get("extracted_data", {})

        new_table_name = extracted.get("table_name")
        if new_table_name:
            maybe_reset_schema(state, new_table_name)
            # Re-read after potential reset so we don't carry stale columns
            raw_schema = state.get("schema_data") or {}
            current_schema = {
                "table_name": new_table_name,
                "columns":    list(raw_schema.get("columns") or [])
            }

        existing_cols = {c["name"]: c for c in current_schema["columns"]}
        for col in extracted.get("columns", []):
            name = col.get("name")
            if not name:
                continue
            if name in existing_cols:
                existing_cols[name].update(col)
            else:
                existing_cols[name] = col

        current_schema["columns"] = list(existing_cols.values())

        # Normalise relation targets
        for col in current_schema["columns"]:
            if col.get("type") == "relation" and col.get("target"):
                col["target"] = _resolve_relation_uid(col["target"])

        state["schema_data"] = current_schema

        # Missing field detection
        missing = []
        if not current_schema.get("table_name"):
            missing.append("table_name")
        if not current_schema.get("columns"):
            missing.append("column_name")
        for col in current_schema["columns"]:
            if not col.get("type"):
                missing.append(f"type for column '{col.get('name')}'")
            if col.get("type") == "enumeration" and not col.get("enum"):
                missing.append(f"enum values for '{col.get('name')}'")
            if col.get("type") == "relation" and not col.get("target"):
                missing.append(f"relation target for '{col.get('name')}'")

        state["missing_fields"] = missing

        print(f"[AddColumnAgent] table_name     : {current_schema.get('table_name')}")
        print(f"[AddColumnAgent] columns        : {current_schema.get('columns')}")
        print(f"[AddColumnAgent] missing_fields : {missing}")
        print(f"[AddColumnAgent] schema_ready   : {not missing}")

        if not missing:
            state["schema_ready"]         = True
            state["interaction_phase"]    = False
            state["active_agent"]         = None
            state["interaction_attempts"] = 0
        else:
            state["schema_ready"]      = False
            state["interaction_phase"] = True
            state["active_agent"]      = "add_column"

    except Exception as e:
        print(f"[AddColumnAgent] Error: {e}")
        state["schema_ready"]   = False
        state["missing_fields"] = ["internal_parsing_error"]

    return state
