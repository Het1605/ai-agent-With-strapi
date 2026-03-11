from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from app.agents.ddl.schema_utils import maybe_reset_schema, format_history
import json
import re


def _resolve_relation_uid(raw_target: str) -> str:
    noise    = r'\b(table|collection|model|entity|db|database|the|a|an)\b'
    cleaned  = re.sub(noise, '', raw_target, flags=re.IGNORECASE).strip()
    singular = re.sub(r'[\s\-_]+', '_', cleaned.lower()).strip('_')
    if singular.endswith('sses') or singular.endswith('ches') or singular.endswith('xes'):
        pass
    elif singular.endswith('ies'):
        singular = singular[:-3] + 'y'
    elif singular.endswith('s') and not singular.endswith('ss'):
        singular = singular[:-1]
    return f"api::{singular}.{singular}"


async def create_table_agent(state: AgentState) -> AgentState:
    """
    CreateTableAgent: extracts full table schema from user input.
    Uses universal prompt template: conversation context, schema_data, missing_fields.
    """
    print("\n----- ENTERING CreateTableAgent -----")

    llm            = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input     = state.get("user_input", "")
    field_registry = state.get("field_registry", {})
    raw_schema     = state.get("schema_data") or {}
    active_agent   = state.get("active_agent") or "create_table"
    missing_fields = state.get("missing_fields") or []

    current_schema = {
        "table_name": raw_schema.get("table_name") or None,
        "columns":    list(raw_schema.get("columns") or [])
    }

    conversation_context = format_history(state, max_turns=4)

    print(f"[CreateTableAgent] user_input  : {user_input}")
    print(f"[CreateTableAgent] table_name  : {current_schema['table_name']}")
    print(f"[CreateTableAgent] columns     : {len(current_schema['columns'])} existing")

    system_prompt = (
        "You are a Database Schema Extraction Agent for an AI database assistant.\n\n"
        "Your job is to extract structured schema instructions from user input and convert them\n"
        "into a precise JSON structure. You analyze conversation context, current schema state,\n"
        "and missing fields to determine what structured information is present.\n\n"
        "You DO NOT generate conversational text. You ONLY return structured JSON.\n\n"
        "BEHAVIOR RULES:\n"
        "1. NEVER hallucinate fields the user did not mention.\n"
        "2. NEVER add constraints (required, unique, default, min, max) unless explicitly stated.\n"
        "3. Use schema_data as the source of truth — merge new info, do not rebuild.\n"
        "4. Only extract what the user actually said.\n"
        "5. If information is missing do not guess — leave it unresolved.\n\n"
        "COLUMN TYPE INFERENCE (infer intelligently):\n"
        "  name/title → string | description/bio → text | age/count/stock → integer\n"
        "  price/amount/total/cost → decimal | email → email | password → password\n"
        "  date/created_at/updated_at → datetime | is_*/active/verified → boolean\n"
        "  If unsure → default to string\n\n"
        "RELATION RULES: If user says 'one X has many Y' → type=relation, relation=oneToMany, target=Y\n"
        "ENUM RULES: If user says 'role (admin, user, guest)' → type=enumeration, enum=[...]\n\n"
        "ENTITY NAME RULES:\n"
        "- Extract the ACTUAL entity noun: 'product', 'order', 'employee'\n"
        "- NEVER use command words as table names: 'create','table','collection','new','add'\n"
        "- If no clear entity name exists → set table_name to null\n\n"
        "MERGING RULES:\n"
        "- Do NOT overwrite existing schema_data columns\n"
        "- Merge new columns into the existing list\n"
        "- A follow-up like 'decimal' or 'price' should be merged with existing partial schema\n\n"
        f"Strapi Field Registry: {json.dumps(field_registry)}\n\n"
        "Return ONLY valid JSON in exactly this format:\n"
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

        # ── Table name + schema reset ────────────────────────────────────
        new_table_name = extracted.get("table_name")
        if new_table_name:
            maybe_reset_schema(state, new_table_name)
            raw_schema = state.get("schema_data") or {}
            current_schema = {
                "table_name": new_table_name,
                "columns":    list(raw_schema.get("columns") or [])
            }

        # ── Merge columns ────────────────────────────────────────────────
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

        # ── Relation UID normalisation ───────────────────────────────────
        for col in current_schema["columns"]:
            if col.get("type") == "relation" and col.get("target"):
                raw_t = col["target"]
                uid   = _resolve_relation_uid(raw_t)
                col["target"] = uid
                print(f"[RelationResolver] {raw_t} → {uid}")

        state["schema_data"] = current_schema

        # ── Missing field detection ──────────────────────────────────────
        # Trust LLM's missing_fields but also enforce Python-level checks
        llm_missing = extracted.get("missing_fields", [])
        missing = list(llm_missing) if llm_missing else []

        if not current_schema.get("table_name"):
            if "table_name" not in missing:
                missing.append("table_name")
        for c in current_schema.get("columns", []):
            if not c.get("type"):
                key = f"data type for column '{c.get('name')}'"
                if key not in missing:
                    missing.append(key)

        state["missing_fields"] = missing

        print(f"[CreateTableAgent] table_name     : {current_schema.get('table_name')}")
        print(f"[CreateTableAgent] columns        : {current_schema.get('columns')}")
        print(f"[CreateTableAgent] missing_fields : {missing}")
        print(f"[CreateTableAgent] schema_ready   : {not missing}")

        if not missing:
            state["schema_ready"]         = True
            state["interaction_phase"]    = False
            state["active_agent"]         = None
            state["interaction_attempts"] = 0
        else:
            state["schema_ready"]      = False
            state["interaction_phase"] = True
            state["active_agent"]      = "create_table"

    except Exception as e:
        print(f"[CreateTableAgent] Error: {e}")
        state["schema_ready"]   = False
        state["missing_fields"] = ["internal_parsing_error"]

    return state
