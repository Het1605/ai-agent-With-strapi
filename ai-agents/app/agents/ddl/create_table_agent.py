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
    print("\n----- ENTERING CreateTableAgent (Senior Architect Upgrade) -----")

    llm            = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input     = state.get("user_input", "")
    full_registry  = state.get("field_registry", {})
    field_registry = full_registry 
    existing_cols_list = state.get("existing_collections")
    if not isinstance(existing_cols_list, list):
        print(f"[CreateTableAgent] Warning: existing_collections is {type(existing_cols_list)}, defaulting to []")
        existing_cols_list = []
    
    raw_schema     = state.get("schema_data") or {}
    active_agent   = state.get("active_agent") or "create_table"

    print("\n[SchemaContext] Existing Collections:")
    if existing_cols_list:
        for col in existing_cols_list:
            print(f" - {col}")
    else:
        print(" (none found)")

    current_schema = {
        "table_name": raw_schema.get("table_name") or None,
        "columns":    list(raw_schema.get("columns") or [])
    }

    conversation_context = format_history(state, max_turns=4)

    print(f"[CreateTableAgent] Input: {user_input}")

    system_prompt = (
        "You are a Senior Database Architect AI designing production-ready systems for Strapi.\n\n"
        "CORE DESIGN PRINCIPLES:\n"
        "1. REALISTIC ENTITIES: Generate complete schemas, not placeholders. Common entities (employee, product, student) MUST have 4-8 fields (e.g. email, phone, status, timestamps handled by Strapi).\n"
        "2. SYSTEM DESIGN: For broad requests (e.g. 'college management'), design a COMPLETE 5-10 table system with normalized relations.\n"
        "3. AUTOMATIC RELATIONS: Infer and link entities logically (order -> customer, enrollment -> student).\n"
        "4. CONSTRAINT INFERENCE: Apply required: true, unique: true, enums (status: ['pending', 'active']), and defaults automatically.\n"
        "5. MULTILINGUAL & SPELLING: Correct errors ('emploee' -> employee) and infer intent from Spanish, French, Hindi, Gujarati, etc.\n"
        "6. NORMALIZATION: Prefer clean, relational data models (e.g. separate addresses into a component or table if appropriate, link objects correctly).\n\n"
        "DATABASE CONTEXT:\n"
        f"Existing Collections in Database: {', '.join(existing_cols_list) if existing_cols_list else 'None'}\n"
        "- NEVER recreate existing tables.\n"
        "- If a relation references an existing table, use it as the target.\n"
        "- Normalize table names to lowercase snake_case.\n\n"
        "STRAPI COMPATIBILITY:\n"
        "- NEVER generate: id, createdAt, updatedAt, publishedAt (Strapi generates these internally).\n"
        "- Use Strapi types: string, text, richtext, email, password, integer, biginteger, float, decimal, datetime, date, time, boolean, enumeration, relation, media, json.\n"
        "- Relation Format: {\"name\": \"...\", \"type\": \"relation\", \"relation\": \"oneToMany|manyToOne|oneToOne|manyToMany\", \"target\": \"target_table\"}\n\n"
        "FIELD CAPABILITIES (Supported Strapi Options):\n"
        f"{json.dumps(state['field_registry'] if state.get('field_registry') else {}, indent=2)}\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- Respond ONLY with valid JSON.\n"
        "- No markdown blocks, no explanation, no comments.\n\n"
        "JSON STRUCTURE:\n"
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
        f"Conversation Context:\n{conversation_context}\n\n"
        f"User Request: {user_input}\n\n"
        f"Existing Collections: {existing_cols_list}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    try:
        clean     = response.content.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(clean)
        raw_tables = extracted.get("tables", [])
        
        tables = []
        for t in raw_tables:
            name = t.get("table_name")
            if name in existing_cols_list:
                print(f"[CreateTableAgent] Skipping existing table: {name}")
                continue
            tables.append(t)

        if not tables:
            print("[CreateTableAgent] All designed tables already exist or none were created.")
            state["execution_result"] = {"message": "All requested entities already exist."}
            state["schema_ready"] = True
            return state

        for table in tables:
            for col in table.get("columns", []):
                if col.get("type") == "relation" and col.get("target"):
                    raw_target = col["target"]
                    uid = _resolve_relation_uid(raw_target)
                    col["target"] = uid
                    print(f"[RelationResolver] {table['table_name']}.{col['name']} -> {uid}")

        state["schema_data"] = {"tables": tables}
        state["schema_ready"]         = True
        state["interaction_phase"]    = False
        state["active_agent"]         = None
        state["interaction_attempts"] = 0

        print(f"[CreateTableAgent] Successfully designed {len(tables)} table(s).")

    except Exception as e:
        print(f"[CreateTableAgent] Design Error: {e}")
        state["execution_error"] = f"Architect Design Error: {str(e)}"

    return state
