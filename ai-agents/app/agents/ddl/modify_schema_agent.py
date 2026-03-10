from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json
import re


def _resolve_relation_uid(raw_target: str) -> str:
    """Normalise relation target to Strapi UID: api::<singular>.<singular>"""
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


async def modify_schema_agent(state: AgentState) -> AgentState:
    """
    ModifySchemaAgent: extracts columns / relations / enums to ADD to an
    existing Strapi collection.  Mirrors CreateTableAgent architecture.

    Routes:
        → interaction_planner  when missing_fields is non-empty
        → query_builder        when schema is complete
    """
    print("\n----- ENTERING ModifySchemaAgent -----")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    user_query     = state.get("user_query", "")
    field_registry = state.get("field_registry", {})

    # Safe-read accumulated schema (never overwrite)
    raw_schema = state.get("schema_data") or {}
    current_schema = {
        "table_name": raw_schema.get("table_name") or None,
        "columns":    list(raw_schema.get("columns") or [])
    }

    system_prompt = (
        "You are a Database Schema Modification Specialist.\n\n"
        f"Available Strapi Field Registry: {json.dumps(field_registry)}\n\n"
        "Instructions:\n"
        "- Extract the TARGET collection name (the table being modified).\n"
        "  This is a domain entity noun: 'product', 'order', 'customer', etc.\n"
        "- NEVER use generic words as collection names: "
        "'collection', 'table', 'new', 'a', 'an', 'the', 'add', 'update', 'field', 'column'.\n"
        "- If no clear target is found, set table_name to null.\n"
        "- Extract ONLY the NEW fields to be added.\n"
        "- Infer column types intelligently:\n"
        "    name / title / label           → string\n"
        "    description / bio / notes      → text\n"
        "    email                          → email\n"
        "    password / secret              → password\n"
        "    price / amount / cost / salary → decimal\n"
        "    age / count / quantity / stock → integer\n"
        "    date / created_at / timestamp  → datetime\n"
        "    is_* / active / verified       → boolean\n"
        "- For ENUM fields: extract the enum values from user message.\n"
        "- For RELATION fields: identify relation type and target collection name.\n"
        "- NEVER add constraints unless user EXPLICITLY stated them.\n\n"
        "Output ONLY valid JSON:\n"
        '{"extracted_data": {"table_name": <string|null>, "columns": [\n'
        '  {"name": "...", "type": "...", '
        '"enum": [...] (optional), "relation": "...", "target": "..." (optional)}\n'
        "]}}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User request: {user_query}")
    ])

    print("Modify_schema response",response)

    try:
        clean = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        extracted = result.get("extracted_data", {})

        # Merge table_name
        if extracted.get("table_name"):
            current_schema["table_name"] = extracted["table_name"]

        # Merge columns (additive, de-duplicated by name)
        existing_cols = {col["name"]: col for col in current_schema["columns"]}
        for new_col in extracted.get("columns", []):
            name = new_col.get("name")
            if not name:
                continue
            if name in existing_cols:
                existing_cols[name].update(new_col)
            else:
                existing_cols[name] = new_col

        current_schema["columns"] = list(existing_cols.values())

        # Normalise all relation targets → Strapi UID
        for col in current_schema["columns"]:
            if col.get("type") == "relation" and col.get("target"):
                raw_t = col["target"]
                uid   = _resolve_relation_uid(raw_t)
                col["target"] = uid
                print(f"[ModifySchemaAgent][RelationResolver] {raw_t!r} → {uid}")

        state["schema_data"] = current_schema

        # Compute missing fields
        missing = []
        if not current_schema.get("table_name"):
            missing.append("table_name")
        for col in current_schema["columns"]:
            if not col.get("type"):
                missing.append(f"data type for column '{col.get('name')}'")
            if col.get("type") == "enumeration" and not col.get("enum"):
                missing.append(f"enum values for '{col.get('name')}'")
            if col.get("type") == "relation" and not col.get("target"):
                missing.append(f"relation target for '{col.get('name')}'")

        state["missing_fields"] = missing

        # Debug logs
        print(f"[ModifySchemaAgent] user_input     : {state.get('user_input')}")
        print(f"[ModifySchemaAgent] table_name     : {current_schema.get('table_name')}")
        print(f"[ModifySchemaAgent] columns        : {current_schema.get('columns')}")
        print(f"[ModifySchemaAgent] missing_fields : {missing}")
        print(f"[ModifySchemaAgent] schema_ready   : {not missing}")

        if not missing:
            state["schema_ready"]         = True
            state["interaction_phase"]    = False
            state["active_agent"]         = None
            state["interaction_attempts"] = 0
            state["debug_info"] = "ModifySchema complete — ready for query building."
        else:
            state["schema_ready"]      = False
            state["interaction_phase"] = True
            state["active_agent"]      = "modify_schema"
            state["debug_info"]        = f"ModifySchema missing: {', '.join(missing)}"

    except Exception as e:
        print(f"[ModifySchemaAgent] Parse error: {e}")
        state["schema_ready"]   = False
        state["missing_fields"] = ["internal_parsing_error"]

    return state
