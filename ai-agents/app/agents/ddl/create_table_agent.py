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
    CreateTableAgent: acts as a Senior Database Architect.
    Automatically designs schemas, infers fields/types/constraints,
    and supports multi-table generation for system designs.
    """
    print("\n----- ENTERING CreateTableAgent (Senior Architect) -----")

    llm            = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input     = state.get("user_input", "")
    full_registry  = state.get("field_registry", {})
    full_registry  = state.get("field_registry", {})
    field_registry = full_registry # backward compatibility if needed, though memory now separates them
    existing_cols_list = state.get("existing_collections", [])
    
    raw_schema     = state.get("schema_data") or {}
    active_agent   = state.get("active_agent") or "create_table"

    print("\n[SchemaContext] Existing Collections:")
    if existing_cols_list:
        for col in existing_cols_list:
            print(col)
    else:
        print("None")

    # We continue to read current_schema for context, but we will output a 'tables' list.
    current_schema = {
        "table_name": raw_schema.get("table_name") or None,
        "columns":    list(raw_schema.get("columns") or [])
    }

    conversation_context = format_history(state, max_turns=4)

    print(f"[CreateTableAgent] user_input: {user_input}")

    system_prompt = (
        "You are a Senior Database Architect AI.\n\n"
        "Your job is to design robust, best-practice database schemas for Strapi based on natural language requests.\n\n"
        "CORE RESPONSIBILITIES:\n"
        "1. AUTOMATIC INFERENCE: If the user provides only a table name (e.g. 'create employee table'), "
        "automatically design a full, realistic schema.\n"
        "2. SYSTEM DESIGN: If the user asks for a system (e.g. 'design ecommerce'), generate ALL necessary tables "
        "and their relationships.\n"
        "3. FIELD INFERENCE: Choose correct types: email, password, decimal, datetime, boolean.\n"
        "4. CONSTRAINT INFERENCE: Add common-sense constraints.\n"
        "5. RELATION DETECTION: Detect and create relations between tables.\n\n"
        "CRITICAL RULES (STRICTLY ENFORCED):\n"
        f"Existing Collections in Database: {', '.join(existing_cols_list) if existing_cols_list else 'None'}\n"
        "1. NEVER generate a table that already exists in the list above.\n"
        "2. If a relation references an existing table, only reference it (e.g. order -> customer).\n"
        "3. Only generate NEW tables explicitly requested or required as new components.\n"
        "4. Do NOT recreate dependency tables if they already exist.\n"
        "5. NEVER generate system fields (id, createdAt, updatedAt, publishedAt).\n"
        "6. Return ONLY valid JSON. No conversational text.\n\n"
        "COLUMN TYPE MAPPING (Strapi types):\n"
        "- string, text, richtext, email, password, integer, biginteger, float, decimal, datetime, date, time, boolean, enumeration, relation, media, json.\n\n"
        "RELATION FORMAT:\n"
        '{"name": "...", "type": "relation", "relation": "oneToMany|manyToOne|oneToOne|manyToMany", "target": "target_table"}\n\n'
        f"Available Strapi Field Options: {json.dumps(field_registry)}\n\n"
        "Return ONLY valid JSON in this exact structure:\n"
        "{\n"
        '  "tables": [\n'
        '    {\n'
        '      "table_name": "...",\n'
        '      "columns": [\n'
        '        {"name": "...", "type": "...", "required": true/false, "unique": true/false, "default": ..., "relation": "...", "target": "..."}\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        "}"
    )

    human_msg = (
        f"[Conversation Context]\n{conversation_context}\n\n"
        f"[Current User Input]\n{user_input}\n\n"
        f"[Current Partial Schema]\n{json.dumps(current_schema)}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    try:
        clean     = response.content.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(clean)
        raw_tables = extracted.get("tables", [])
        
        # ── Filter out existing tables (Fail-safe) ───────────────────────
        tables = []
        for t in raw_tables:
            name = t.get("table_name")
            if name in existing_cols_list:
                print(f"[CreateTableAgent] Skipped existing table: {name}")
                continue
            tables.append(t)

        if not tables:
            print("[CreateTableAgent] Warning: No NEW tables generated by LLM.")
            # Fallback only if we absoluteley must
            if current_schema.get("table_name") and current_schema["table_name"] not in existing_cols_list:
                tables = [current_schema]
            else:
                # If everything exists, we just success out or return warning
                state["execution_result"] = {"message": "All requested tables already exist."}
                state["schema_ready"] = True
                return state

        # ── Normalise relations and UIDs for all tables ───────────────────
        for table in tables:
            for col in table.get("columns", []):
                if col.get("type") == "relation" and col.get("target"):
                    raw_target = col["target"]
                    uid = _resolve_relation_uid(raw_target)
                    col["target"] = uid
                    print(f"[RelationResolver] {table['table_name']}.{col['name']} -> {uid}")

        # Update state with the designed tables
        state["schema_data"] = {"tables": tables}
        
        state["schema_ready"]         = True
        state["missing_fields"]       = []
        state["interaction_phase"]    = False
        state["active_agent"]         = None
        state["interaction_attempts"] = 0

        print(f"[CreateTableAgent] Designed {len(tables)} new table(s) autonomously.")

    except Exception as e:
        print(f"[CreateTableAgent] Error: {e}")
        state["execution_error"] = f"Senior Architect Error: {str(e)}"

    return state
