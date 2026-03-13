from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from app.agents.ddl.schema_utils import maybe_reset_schema, format_history
import json

async def add_column_agent(state: AgentState) -> AgentState:
    """
    AddColumnAgent: extracts NEW fields to add to an existing collection.
    Uses universal prompt template with conversation context injection.
    """
    print("\n----- ENTERING AddColumnAgent -----")

    llm            = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input     = state.get("user_input", "")
    field_registry = state.get("field_registry", {})
    modify_op      = state.get("modify_operation") or {}
    raw_schema     = state.get("schema_data") or {}
    active_agent   = state.get("active_agent") or "add_column"
    missing_fields = state.get("missing_fields") or []

    current_schema = {
        "table_name": modify_op.get("target_table") or raw_schema.get("table_name") or None,
        "columns":    list(raw_schema.get("columns") or [])
    }

    conversation_context = format_history(state, max_turns=4)

    print(f"[AddColumnAgent] user_input   : {user_input}")
    print(f"[AddColumnAgent] seeded table : {current_schema['table_name']}")

    system_prompt = (
        "You are a Database Schema Extraction Agent for an AI database assistant.\n\n"
        "Your job is to extract NEW FIELD definitions to add to an existing collection.\n"
        "You DO NOT generate conversational text. You ONLY return structured JSON.\n\n"
        "BEHAVIOR RULES:\n"
        "1. NEVER hallucinate fields the user did not mention.\n"
        "2. NEVER add constraints (required, unique, default, min, max) unless explicitly stated.\n"
        "3. Use schema_data as the source of truth — merge new columns, do not rebuild.\n"
        "4. Only extract what the user actually said.\n"
        "5. If information is missing, do not guess — leave it unresolved.\n\n"
        "ENTITY NAME RULES:\n"
        "- Extract the ACTUAL entity noun: 'product', 'order', 'employee'\n"
        "- NEVER use: 'collection','table','column','field','add','new' as table names\n\n"
        "COLUMN TYPE INFERENCE:\n"
        "  name/title → string | description/bio → text | age/count/stock/qty → integer\n"
        "  price/amount/total/cost/salary → decimal | email → email | password → password\n"
        "  date/created_at/updated_at → datetime | is_*/active/verified → boolean\n"
        "  If unsure → string\n\n"
        "RELATION RULES: 'one X has many Y' → type=relation, relation=oneToMany, target=Y\n"
        "ENUM RULES: 'role (admin, user, guest)' → type=enumeration, enum=['admin','user','guest']\n\n"
        "MERGING RULES:\n"
        "- Do NOT overwrite existing columns\n"
        "- Only add the NEW columns the user mentioned\n"
        "- A follow-up like 'decimal' resolves the type for a previously unnamed column\n\n"
        f"Strapi Field Registry: {json.dumps(field_registry)}\n\n"
        "Return ONLY valid JSON:\n"
        '{"table_name": <str|null>, "columns": [...], "updates": {}, "missing_fields": []}'
    )

    human_msg = (
        f"[Conversation Context]\n{conversation_context}\n\n"
        f"[Current User Input]\n{user_input}\n\n"
        f"[Current Schema State]\n{json.dumps(current_schema)}\n\n"
        f"[Missing Fields]\n{json.dumps(missing_fields)}\n\n"
        f"[Active Agent]\n{active_agent}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    try:
        clean     = response.content.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(clean)

        new_table_name = extracted.get("table_name")
        if new_table_name:
            maybe_reset_schema(state, new_table_name)
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

        # No more hardcoded naming logic. Relation targets are passed raw to QueryBuilder.
        state["schema_data"] = current_schema

        # Missing field detection — combine LLM suggestion with Python checks
        llm_missing = extracted.get("missing_fields", [])
        missing = list(llm_missing) if llm_missing else []

        if not current_schema.get("table_name") and "table_name" not in missing:
            missing.append("table_name")
        if not current_schema.get("columns") and "column_name" not in missing:
            missing.append("column_name")
        for col in current_schema["columns"]:
            if not col.get("type"):
                key = f"type for column '{col.get('name')}'"
                if key not in missing:
                    missing.append(key)
            if col.get("type") == "enumeration" and not col.get("enum"):
                key = f"enum values for '{col.get('name')}'"
                if key not in missing:
                    missing.append(key)
            if col.get("type") == "relation" and not col.get("target"):
                key = f"relation target for '{col.get('name')}'"
                if key not in missing:
                    missing.append(key)

        state["missing_fields"] = missing

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
