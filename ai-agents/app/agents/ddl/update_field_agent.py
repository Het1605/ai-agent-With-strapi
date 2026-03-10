from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
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
        "You are a Database Field Configuration Specialist.\n\n"
        "Instructions:\n"
        "- Extract the TARGET collection name (entity noun: 'product', 'customer', etc.).\n"
        "- Extract the TARGET field name to be updated.\n"
        "- Extract the updates to apply. Possible update keys:\n"
        "    required: true/false\n"
        "    unique: true/false\n"
        "    private: true/false\n"
        "    default: <value>\n"
        "    new_name: <string>  (for rename operations — you must include this key when user says rename)\n"
        "- NEVER make up values. Only extract what user explicitly states.\n"
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

        if extracted.get("table_name"):
            current_schema["table_name"] = extracted["table_name"]
        if extracted.get("field_name"):
            current_schema["field_name"] = extracted["field_name"]
        if extracted.get("updates"):
            current_schema["updates"] = {**current_schema["updates"], **extracted["updates"]}

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
