from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from app.agents.ddl.schema_utils import maybe_reset_schema
import json


async def update_field_agent(state: AgentState) -> AgentState:
    """
    UpdateFieldAgent: modifies settings of an existing field in a collection.
    Examples: make price required, make email unique, rename field price to product_price.

    Routes → interaction_planner (missing info) or → query_builder (complete).
    """
    print("\n----- ENTERING UpdateFieldAgent -----")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    user_input = state.get("user_input", "")
    raw_schema = state.get("schema_data") or {}
    current_schema = {
        "table_name": raw_schema.get("table_name") or None,
        "field_name": raw_schema.get("field_name") or None,
        "updates":    raw_schema.get("updates") or {},
    }

    print(f"[UpdateFieldAgent] user_input : {user_input}")

    system_prompt = (
        "You are a strict database schema modification agent.\n\n"
        "Your task is to extract field updates from the user's request.\n\n"
        "CRITICAL RULES:\n"
        "1. NEVER add settings the user did NOT explicitly mention.\n"
        "2. NEVER infer or assume constraints such as unique, required, min, max, "
        "private, searchable, configurable unless the user explicitly requested them.\n"
        "3. If the user asks to set a default value → return ONLY {'default': <value>}.\n"
        "4. If the user asks to make a field required → return ONLY {'required': true}.\n"
        "5. If the user asks to make a field unique → return ONLY {'unique': true}.\n"
        "6. For rename requests → return ONLY {'new_name': '<new_name>'}.\n"
        "7. Do NOT combine multiple settings automatically.\n\n"
        "Example — correct:\n"
        "  User: set default value of rating to 5.0\n"
        "  Output updates: {\"default\": 5.0}\n\n"
        "Example — WRONG (never do this):\n"
        "  Output updates: {\"default\": 5.0, \"unique\": true}\n\n"
        "- Extract the TARGET collection name (entity noun: 'product', 'customer', etc.).\n"
        "- Extract the TARGET field name to be updated.\n"
        "- If table_name, field_name, or updates cannot be determined, set to null/empty.\n\n"
        "Output ONLY valid JSON:\n"
        '{"extracted_data": {"table_name": <str|null>, "field_name": <str|null>, "updates": {}}}'
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
            current_schema["table_name"] = new_table_name
        if extracted.get("field_name"):
            current_schema["field_name"] = extracted["field_name"]
        if extracted.get("updates"):
            # Allowlist filter — only store keys the LLM explicitly returned.
            # This prevents hallucinated constraints from sneaking through.
            _ALLOWED_UPDATE_KEYS = {
                "default", "required", "unique", "min", "max",
                "enum", "new_name", "private", "searchable"
            }
            safe_updates = {
                k: v for k, v in extracted["updates"].items()
                if k in _ALLOWED_UPDATE_KEYS
            }
            current_schema["updates"] = {**current_schema["updates"], **safe_updates}
            if len(safe_updates) != len(extracted["updates"]):
                dropped = set(extracted["updates"]) - set(safe_updates)
                print(f"[UpdateFieldAgent] Dropped non-allowed keys: {dropped}")

        state["schema_data"] = current_schema

        # Missing field detection
        missing = []
        if not current_schema.get("table_name"):
            missing.append("table_name")
        if not current_schema.get("field_name"):
            missing.append("field_name")
        if not current_schema.get("updates"):
            missing.append("update_action (e.g. required, unique, new_name)")

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
