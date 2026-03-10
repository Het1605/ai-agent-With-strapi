from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json


async def update_collection_agent(state: AgentState) -> AgentState:
    """
    UpdateCollectionAgent: handles collection-level changes.
    Supports:
      - Renaming displayName
      - Deleting the entire collection (data.delete = true)

    Routes → interaction_planner (missing info) or → query_builder (complete).
    """
    print("\n----- ENTERING UpdateCollectionAgent -----")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    user_input = state.get("user_input", "")
    raw_schema = state.get("schema_data") or {}
    current_schema = {
        "table_name": raw_schema.get("table_name") or None,
    }

    print(f"[UpdateCollectionAgent] user_input : {user_input}")

    system_prompt = (
        "You are a Collection Settings Specialist in a multi-agent database system.\n\n"
        "Your job is to extract the intent for a collection-level modification.\n\n"
        "Supported intents:\n"
        "1. DELETE collection — user says 'delete collection X', 'drop table X', 'remove collection X'\n"
        "2. RENAME / UPDATE displayName — user says 'rename X to Y', 'update display name of X to Y'\n\n"
        "Instructions:\n"
        "- Extract the TARGET collection name (entity noun: 'product', 'order', etc.).\n"
        "- NEVER treat generic words as names: 'collection', 'table', 'the', 'rename', 'delete'.\n"
        "- If no collection name found, set table_name to null.\n"
        "- If delete intent: set delete to true.\n"
        "- If rename intent: set new_display_name to the new name.\n\n"
        "Output ONLY valid JSON:\n"
        '{"extracted_data": {"table_name": <str|null>, "delete": <bool>, "new_display_name": <str|null>}}'
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

        # Build the operation-specific data payload
        if extracted.get("delete") == True:
            current_schema["delete"] = True
            current_schema["new_display_name"] = None
        else:
            current_schema["delete"] = False
            current_schema["new_display_name"] = extracted.get("new_display_name")

        state["schema_data"] = current_schema

        # Missing field detection
        missing = []
        if not current_schema.get("table_name"):
            missing.append("table_name")
        if not current_schema.get("delete") and not current_schema.get("new_display_name"):
            missing.append("operation detail (new display name or delete confirmation)")

        state["missing_fields"] = missing

        print(f"[UpdateCollectionAgent] table_name      : {current_schema.get('table_name')}")
        print(f"[UpdateCollectionAgent] delete          : {current_schema.get('delete')}")
        print(f"[UpdateCollectionAgent] new_display_name: {current_schema.get('new_display_name')}")
        print(f"[UpdateCollectionAgent] missing_fields  : {missing}")
        print(f"[UpdateCollectionAgent] schema_ready    : {not missing}")

        if not missing:
            state["schema_ready"]         = True
            state["interaction_phase"]    = False
            state["active_agent"]         = None
            state["interaction_attempts"] = 0
        else:
            state["schema_ready"]      = False
            state["interaction_phase"] = True
            state["active_agent"]      = "update_collection"

    except Exception as e:
        print(f"[UpdateCollectionAgent] Error: {e}")
        state["schema_ready"]   = False
        state["missing_fields"] = ["internal_parsing_error"]

    return state
