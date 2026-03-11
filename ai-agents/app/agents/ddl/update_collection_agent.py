from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from app.agents.ddl.schema_utils import maybe_reset_schema, format_history
import json


async def update_collection_agent(state: AgentState) -> AgentState:
    """
    UpdateCollectionAgent: handles collection-level changes.
    Supports renaming displayName and deleting the entire collection.
    Uses universal prompt template with conversation context injection.
    """
    print("\n----- ENTERING UpdateCollectionAgent -----")

    llm          = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input   = state.get("user_input", "")
    modify_op    = state.get("modify_operation") or {}
    active_agent = state.get("active_agent") or "update_collection"
    missing_fields = state.get("missing_fields") or []

    # Always start fresh — never carry previous collection state
    current_schema = {
        "table_name": modify_op.get("target_table") or None,
    }

    conversation_context = format_history(state, max_turns=4)

    print(f"[UpdateCollectionAgent] user_input    : {user_input}")
    print(f"[UpdateCollectionAgent] seeded table  : {current_schema['table_name']}")

    system_prompt = (
        "You are a Database Schema Extraction Agent for an AI database assistant.\n\n"
        "Your job is to extract COLLECTION-LEVEL modification instructions from user input.\n"
        "You DO NOT generate conversational text. You ONLY return structured JSON.\n\n"
        "BEHAVIOR RULES:\n"
        "1. NEVER add settings the user did not explicitly mention.\n"
        "2. Extract the TARGET collection name (entity noun: 'product', 'order', etc.)\n"
        "   NEVER use: 'collection','table','the','rename','delete' as the name.\n"
        "3. If collection name cannot be identified from input or context → set table_name to null.\n\n"
        "SUPPORTED OPERATIONS:\n"
        "  DELETE collection: user says 'delete X', 'drop X', 'remove collection X'\n"
        "    → set updates.delete = true\n"
        "  RENAME display name: user says 'rename X to Y', 'update display name of X to Y'\n"
        "    → set updates.new_display_name = 'Y'\n\n"
        "CONTEXT USE: Use conversation history to resolve collection name if not stated explicitly.\n\n"
        "Return ONLY valid JSON:\n"
        '{"table_name": <str|null>, "columns": [], "updates": {"delete": false, "new_display_name": null}, "missing_fields": []}'
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
        print(f"[UpdateCollectionAgent] LLM raw: {clean}")
        extracted = json.loads(clean)

        new_table_name = extracted.get("table_name")
        if new_table_name:
            maybe_reset_schema(state, new_table_name)
            current_schema["table_name"] = new_table_name

        updates = extracted.get("updates") or {}
        if updates.get("delete") == True:
            current_schema["delete"]           = True
            current_schema["new_display_name"] = None
        else:
            current_schema["delete"]           = False
            current_schema["new_display_name"] = updates.get("new_display_name")

        state["schema_data"] = current_schema

        # Missing field detection
        llm_missing = extracted.get("missing_fields") or []
        missing = list(llm_missing)

        if not current_schema.get("table_name") and "table_name" not in missing:
            missing.append("table_name")
        if (not current_schema.get("delete") and not current_schema.get("new_display_name")
                and "operation detail (new display name or delete confirmation)" not in missing):
            missing.append("operation detail (new display name or delete confirmation)")

        state["missing_fields"] = missing

        print(f"[UpdateCollectionAgent] table_name      : {current_schema.get('table_name')}")
        print(f"[UpdateCollectionAgent] delete          : {current_schema.get('delete')}")
        print(f"[UpdateCollectionAgent] new_display_name: {current_schema.get('new_display_name')}")
        print(f"[UpdateCollectionAgent] missing_fields  : {missing}")

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
