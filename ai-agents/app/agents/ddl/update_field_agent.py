from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from app.agents.ddl.schema_utils import maybe_reset_schema
import json


_ALLOWED_UPDATE_KEYS = frozenset({
    "default", "required", "unique", "min", "max",
    "enum", "new_name", "private", "searchable"
})


async def update_field_agent(state: AgentState) -> AgentState:
    """
    UpdateFieldAgent: modifies settings of an existing field in a collection.

    Routes → interaction_planner (missing info) or → query_builder (complete).

    Key design rule: `updates` is ALWAYS reset to {} at the start of each call.
    Previous updates are NEVER carried forward — this prevents cross-request leakage.
    """
    print("\n----- ENTERING UpdateFieldAgent -----")

    llm        = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input = state.get("user_input", "")

    # Seed from ModifySchemaAgent's pre-extracted context if available
    modify_op   = state.get("modify_operation") or {}
    raw_schema  = state.get("schema_data") or {}

    # CRITICAL: never carry forward the old updates dict — start fresh every call
    current_schema = {
        "table_name": modify_op.get("target_table") or raw_schema.get("table_name") or None,
        "field_name": modify_op.get("target_field") or raw_schema.get("field_name") or None,
        "updates":    {},   # always empty — prevents cross-request leakage
    }

    print(f"[UpdateFieldAgent] user_input    : {user_input}")
    print(f"[UpdateFieldAgent] seeded table  : {current_schema['table_name']}")
    print(f"[UpdateFieldAgent] seeded field  : {current_schema['field_name']}")

    system_prompt = (
        "You are a strict database schema modification agent.\n\n"
        "Your task: extract ONLY the field update the user explicitly requested.\n\n"
        "CRITICAL RULES:\n"
        "1. NEVER add settings the user did NOT mention.\n"
        "2. NEVER infer or assume unique, required, min, max, private, searchable, configurable.\n"
        "3. set default value → return ONLY {\"default\": <value>}\n"
        "4. make required     → return ONLY {\"required\": true}\n"
        "5. make unique       → return ONLY {\"unique\": true}\n"
        "6. rename field      → return ONLY {\"new_name\": \"<new_name>\"}\n"
        "7. NEVER combine multiple updates unless the user explicitly requested both.\n\n"
        "Examples (correct):\n"
        "  'set default value of rating to 5'           → updates: {\"default\": 5}\n"
        "  'rename column email to email_address'       → updates: {\"new_name\": \"email_address\"}\n"
        "  'make price required'                        → updates: {\"required\": true}\n\n"
        "Examples (WRONG — never do this):\n"
        "  'set default value of rating to 5'           → updates: {\"default\": 5, \"unique\": true}\n\n"
        "Output ONLY valid JSON:\n"
        '{"extracted_data": {"table_name": <str|null>, "field_name": <str|null>, "updates": {}}}'
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User request: {user_input}")
    ])

    try:
        clean = response.content.replace("```json", "").replace("```", "").strip()
        print(f"[UpdateFieldAgent] LLM raw: {clean}")
        extracted = json.loads(clean).get("extracted_data", {})

        new_table_name = extracted.get("table_name")
        if new_table_name:
            maybe_reset_schema(state, new_table_name)
            current_schema["table_name"] = new_table_name

        if extracted.get("field_name"):
            current_schema["field_name"] = extracted["field_name"]

        if extracted.get("updates"):
            # Allowlist filter — drop any key the LLM invented outside the permitted set
            safe_updates = {
                k: v for k, v in extracted["updates"].items()
                if k in _ALLOWED_UPDATE_KEYS
            }
            if len(safe_updates) != len(extracted["updates"]):
                dropped = set(extracted["updates"]) - set(safe_updates)
                print(f"[UpdateFieldAgent] Dropped non-allowed keys: {dropped}")
            # Replace updates entirely — never merge with old state
            current_schema["updates"] = safe_updates

        state["schema_data"] = current_schema

        # Missing field detection
        missing = []
        if not current_schema.get("table_name"):
            missing.append("table_name")
        if not current_schema.get("field_name"):
            missing.append("field_name")
        if not current_schema.get("updates"):
            missing.append("update_action (e.g. required, unique, new_name, default)")

        state["missing_fields"] = missing

        print(f"[UpdateFieldAgent] table_name     : {current_schema.get('table_name')}")
        print(f"[UpdateFieldAgent] field_name     : {current_schema.get('field_name')}")
        print(f"[UpdateFieldAgent] updates        : {current_schema.get('updates')}")
        print(f"[UpdateFieldAgent] missing_fields : {missing}")
        print(f"[UpdateFieldAgent] schema_ready   : {not missing}")

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
