from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from app.agents.ddl.schema_utils import maybe_reset_schema
import json
import re


def _resolve_relation_uid(raw_target: str) -> str:
    """
    Normalise a raw relation target string to the Strapi content-type UID format:
        api::<singularName>.<singularName>

    Examples
    --------
    'Order'         → 'api::order.order'
    'orders'        → 'api::order.order'
    'orders table'  → 'api::order.order'
    'order collection' → 'api::order.order'
    """
    # Strip common noise words
    noise = r'\b(table|collection|model|entity|db|database|the|a|an)\b'
    cleaned = re.sub(noise, '', raw_target, flags=re.IGNORECASE).strip()
    # Lowercase and collapse whitespace / underscores / hyphens to single token
    singular = re.sub(r'[\s\-_]+', '_', cleaned.lower()).strip('_')
    # Remove trailing 's' to convert simple plurals to singular
    # e.g. 'orders' → 'order', 'customers' → 'customer'
    # Avoid stripping 's' from words that are naturally singular (e.g. 'address', 'class')
    if singular.endswith('sses') or singular.endswith('ches') or singular.endswith('xes'):
        pass  # already a natural singular, leave alone
    elif singular.endswith('ies'):
        singular = singular[:-3] + 'y'   # categories → category
    elif singular.endswith('s') and not singular.endswith('ss'):
        singular = singular[:-1]
    uid = f"api::{singular}.{singular}"
    return uid

async def create_table_agent(state: AgentState) -> AgentState:
    """
    CreateTableAgent: AI agent that uses LLM reasoning to validate and extract table schema.
    It ensures all column properties (min, max, required, etc.) are captured.
    """
    print("\n----- ENTERING CreateTableAgent (AI Reasoning) -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    user_query = state.get("user_query", "")
    # Safely read existing schema without overwriting it
    raw_schema = state.get("schema_data") or {}
    print("raw_schema",raw_schema)

    current_schema = {
        "table_name": raw_schema.get("table_name") or None,
        "columns": list(raw_schema.get("columns") or [])
    }

    print("current_schema",current_schema)

    field_registry = state.get("field_registry", {})

    print("field_registry:", field_registry)
    
    system_prompt = (
        "You are a Database Schema Specialist. Your role is to extract schema information from the user's message.\n\n"
        f"Available Strapi Field Registry: {json.dumps(field_registry)}\n\n"
        "Instructions:\n"
        "- Extract the ACTUAL ENTITY (table) name from the user message. This is a domain noun — e.g. 'order', 'product', 'employee', 'invoice'.\n"
        "- CRITICAL: Do NOT treat generic command words as table names. The following words are NEVER a table name: 'collection', 'table', 'new', 'a', 'an', 'the', 'create', 'make', 'add', 'database', 'db'.\n"
        "- If no clear ENTITY name is present, set table_name to null.\n"
        "- Extract column definitions only if the user explicitly lists fields or attributes.\n"
        "- Intelligently INFER column types from names (e.g. name→string, age→integer, price→decimal/float, email→email, date→date, is_active→boolean). Use the field registry for Strapi-compatible types.\n"
        "- NEVER add constraints (required, unique, min, max, default) unless the user EXPLICITLY stated them.\n"
        "- Output ONLY a valid JSON object: {\"extracted_data\": {\"table_name\": <string or null>, \"columns\": [...]}}\n"
        "Output ONLY valid JSON."
    )
    
    human_msg = (
        f"Current User Message: {user_query}\n"
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])
    
    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_content)
        extracted = result.get("extracted_data", {})
        
        # 1. Read existing state
        # 2. Extract new info (done by LLM)
        # 3. Merge new fragment into schema_data
        # 4. Preserve previously collected fields
        
        # Merge table name — detect table change and reset before merging
        new_table_name = extracted.get("table_name")
        if new_table_name:
            maybe_reset_schema(state, new_table_name)
            # Re-read after potential reset
            raw_schema = state.get("schema_data") or {}
            current_schema = {
                "table_name": new_table_name,
                "columns":    list(raw_schema.get("columns") or [])
            }

        # Merge columns
        existing_cols = {col["name"]: col for col in current_schema.get("columns", [])}
        for new_col in extracted.get("columns", []):
            name = new_col.get("name")
            if not name:
                continue
            if name in existing_cols:
                existing_cols[name].update(new_col)
            else:
                existing_cols[name] = new_col
                
        current_schema["columns"] = list(existing_cols.values())

        # ── Relation target normalization ────────────────────────────
        # Strapi requires UID format: api::<singular>.<singular>
        # Replace any plain target name on relation-type columns.
        for col in current_schema["columns"]:
            if col.get("type") == "relation" and col.get("target"):
                raw_target = col["target"]
                uid = _resolve_relation_uid(raw_target)
                col["target"] = uid
                print(f"[RelationResolver]")
                print(f"  raw_target        : {raw_target}")
                print(f"  normalized_target : {uid.split('::')[1].split('.')[0]}")
                print(f"  uid               : {uid}")
        # ─────────────────────────────────────────────────────────────

        state["schema_data"] = current_schema
        
        # Compute missing fields dynamically.
        # RULE: table_name is the ONLY mandatory field.
        # Columns are optional — Strapi allows adding fields later.
        # Only flag a column if the user provided it but the type could not be inferred.
        missing = []
        if not current_schema.get("table_name"):
            missing.append("table_name")
        for c in current_schema.get("columns", []):
            if "type" not in c or not c["type"]:
                missing.append(f"data type for column '{c.get('name')}'")

        state["missing_fields"] = missing

        # ── Debug logs ──────────────────────────────────────────────
        print(f"[CreateTableAgent] user_input     : {state.get('user_input')}")
        print(f"[CreateTableAgent] schema_data    : {json.dumps(current_schema)}")
        print(f"[CreateTableAgent] table_name     : {current_schema.get('table_name')}")
        print(f"[CreateTableAgent] columns        : {current_schema.get('columns', [])}")
        print(f"[CreateTableAgent] missing_fields : {missing}")
        print(f"[CreateTableAgent] schema_ready   : {not missing}")
        # ────────────────────────────────────────────────────────────

        if not missing:
            state["schema_ready"] = True
            state["interaction_phase"] = False
            state["active_agent"] = None
            state["interaction_attempts"] = 0
            state["debug_info"] = "Schema is complete and ready for query building."
        else:
            state["schema_ready"] = False
            state["interaction_phase"] = True
            state["active_agent"] = "create_table"
            state["debug_info"] = f"Missing schema information detected: {', '.join(missing)}"
            
    except Exception as e:
        print(f"Error in CreateTableAgent reasoning: {e}")
        state["schema_ready"] = False
        state["missing_fields"] = ["internal_parsing_error"]
        
    return state
