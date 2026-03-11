from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from app.agents.ddl.schema_utils import maybe_reset_schema, format_history
import json


_ALLOWED_UPDATE_KEYS = frozenset({
    "default", "required", "unique", "min", "max",
    "enum", "new_name", "private", "searchable"
})


async def update_field_agent(state: AgentState) -> AgentState:
    """
    UpdateFieldAgent: modifies settings of an existing field.
    Uses universal prompt template with conversation context injection.
    CRITICAL: updates dict is always reset to {} — never carried forward.
    """
    print("\n----- ENTERING UpdateFieldAgent -----")

    llm        = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input = state.get("user_input", "")
    modify_op  = state.get("modify_operation") or {}
    active_agent = state.get("active_agent") or "update_field"
    missing_fields = state.get("missing_fields") or []
    raw_schema = state.get("schema_data") or {}

    # CRITICAL: never carry forward the old updates dict
    current_schema = {
        "table_name": modify_op.get("target_table") or raw_schema.get("table_name") or None,
        "field_name": modify_op.get("target_field") or raw_schema.get("field_name") or None,
        "updates":    {},   # always empty — prevents cross-request leakage
    }

    conversation_context = format_history(state, max_turns=4)

    print(f"[UpdateFieldAgent] user_input   : {user_input}")
    print(f"[UpdateFieldAgent] seeded table : {current_schema['table_name']}")
    print(f"[UpdateFieldAgent] seeded field : {current_schema['field_name']}")

    system_prompt = (
        "You are a strict Database Schema Extraction Agent for an AI database assistant.\n\n"
        "Your job is to extract FIELD-LEVEL modification instructions from user input.\n"
        "You DO NOT generate conversational text. You ONLY return structured JSON.\n\n"
        "CRITICAL RULES:\n"
        "1. NEVER add settings the user did NOT explicitly mention.\n"
        "2. NEVER infer or assume: unique, required, min, max, private, searchable.\n"
        "3. set default value     → updates: {\"default\": <value>} ONLY\n"
        "4. make required         → updates: {\"required\": true} ONLY\n"
        "5. make unique           → updates: {\"unique\": true} ONLY\n"
        "6. rename field          → updates: {\"new_name\": \"<name>\"} ONLY\n"
        "7. NEVER combine multiple updates unless user explicitly requested both.\n\n"
        "ENTITY NAME RULES:\n"
        "- Extract the TARGET collection name (entity noun only)\n"
        "- Extract the TARGET field name to modify\n"
        "- Use conversation history to resolve ambiguous references ('it', 'that field', 'the column')\n\n"
        "Examples (correct):\n"
        "  'set default of rating to 5'             → updates: {\"default\": 5}\n"
        "  'rename email to email_address'           → updates: {\"new_name\": \"email_address\"}\n"
        "  'make price required'                     → updates: {\"required\": true}\n\n"
        "Return ONLY valid JSON:\n"
        '{"table_name": <str|null>, "field_name": <str|null>, "columns": [], "updates": {}, "missing_fields": []}'
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
        clean = response.content.replace("```json", "").replace("```", "").strip()
        print(f"[UpdateFieldAgent] LLM raw: {clean}")
        extracted = json.loads(clean)

        new_table_name = extracted.get("table_name")
        if new_table_name:
            maybe_reset_schema(state, new_table_name)
            current_schema["table_name"] = new_table_name

        if extracted.get("field_name"):
            current_schema["field_name"] = extracted["field_name"]

        if extracted.get("updates"):
            # Allowlist filter + replace (never merge with old)
            safe_updates = {
                k: v for k, v in extracted["updates"].items()
                if k in _ALLOWED_UPDATE_KEYS
            }
            dropped = set(extracted["updates"]) - set(safe_updates)
            if dropped:
                print(f"[UpdateFieldAgent] Dropped non-allowed keys: {dropped}")
            current_schema["updates"] = safe_updates

        state["schema_data"] = current_schema

        # Missing field detection
        llm_missing = extracted.get("missing_fields") or []
        missing = list(llm_missing)

        if not current_schema.get("table_name") and "table_name" not in missing:
            missing.append("table_name")
        if not current_schema.get("field_name") and "field_name" not in missing:
            missing.append("field_name")
        if not current_schema.get("updates") and "update_action (e.g. required, unique, new_name, default)" not in missing:
            missing.append("update_action (e.g. required, unique, new_name, default)")

        state["missing_fields"] = missing

        print(f"[UpdateFieldAgent] table_name     : {current_schema.get('table_name')}")
        print(f"[UpdateFieldAgent] field_name     : {current_schema.get('field_name')}")
        print(f"[UpdateFieldAgent] updates        : {current_schema.get('updates')}")
        print(f"[UpdateFieldAgent] missing_fields : {missing}")

        if not missing:
            state["schema_ready"]         = True
            state["interaction_phase"]    = False
            state["active_agent"]         = None
            state["interaction_attempts"] = 0
        else:
            state["schema_ready"]      = False
            state["interaction_phase"] = True
            state["active_agent"]      = "update_field"

    except Exception as e:
        print(f"[UpdateFieldAgent] Error: {e}")
        state["schema_ready"]   = False
        state["missing_fields"] = ["internal_parsing_error"]

    return state
