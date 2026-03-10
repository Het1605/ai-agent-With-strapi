from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from app.agents.ddl.schema_utils import maybe_reset_schema
import json


async def delete_field_agent(state: AgentState) -> AgentState:
    """
    DeleteFieldAgent: identifies a specific field to remove from an existing collection.
    Examples: delete column price from product, remove email from customer.

    Routes → interaction_planner (missing info) or → query_builder (complete).
    """
    print("\n----- ENTERING DeleteFieldAgent -----")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    user_input = state.get("user_input", "")
    raw_schema = state.get("schema_data") or {}
    current_schema = {
        "table_name": raw_schema.get("table_name") or None,
        "field_name": raw_schema.get("field_name") or None,
    }

    print(f"[DeleteFieldAgent] user_input : {user_input}")

    system_prompt = (
        "You are a Database Field Deletion Specialist.\n\n"
        "Instructions:\n"
        "- Extract the TARGET collection name (entity noun: 'product', 'customer', etc.).\n"
        "- Extract the TARGET field name to be deleted.\n"
        "- NEVER treat generic words as names: 'collection', 'table', 'field', 'column', 'delete', 'remove'.\n"
        "- If table_name or field_name cannot be clearly identified, set to null.\n\n"
        "Output ONLY valid JSON:\n"
        '{"extracted_data": {"table_name": <str|null>, "field_name": <str|null>}}'
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

        state["schema_data"] = current_schema

        # Missing field detection
        missing = []
        if not current_schema.get("table_name"):
            missing.append("table_name")
        if not current_schema.get("field_name"):
            missing.append("field_name")

        state["missing_fields"] = missing

        print(f"[DeleteFieldAgent] table_name     : {current_schema.get('table_name')}")
        print(f"[DeleteFieldAgent] field_name     : {current_schema.get('field_name')}")
        print(f"[DeleteFieldAgent] missing_fields : {missing}")
        print(f"[DeleteFieldAgent] schema_ready   : {not missing}")

        if not missing:
            state["schema_ready"]         = True
            state["interaction_phase"]    = False
            state["active_agent"]         = None
            state["interaction_attempts"] = 0
        else:
            state["schema_ready"]      = False
            state["interaction_phase"] = True
            state["active_agent"]      = "delete_field"

    except Exception as e:
        print(f"[DeleteFieldAgent] Error: {e}")
        state["schema_ready"]   = False
        state["missing_fields"] = ["internal_parsing_error"]

    return state
