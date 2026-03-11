from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from app.agents.ddl.schema_utils import maybe_reset_schema, format_history
import json


async def delete_field_agent(state: AgentState) -> AgentState:
    """
    DeleteFieldAgent: identifies the field to remove from an existing collection.
    Uses universal prompt template with conversation context injection.
    """
    print("\n----- ENTERING DeleteFieldAgent -----")

    llm        = ChatOpenAI(model="gpt-4o", temperature=0)
    user_input = state.get("user_input", "")
    modify_op  = state.get("modify_operation") or {}
    active_agent = state.get("active_agent") or "delete_field"
    missing_fields = state.get("missing_fields") or []

    # Always start fresh — never carry previous field state
    current_schema = {
        "table_name": modify_op.get("target_table") or None,
        "field_name": modify_op.get("target_field") or None,
    }

    conversation_context = format_history(state, max_turns=4)

    print(f"[DeleteFieldAgent] user_input    : {user_input}")
    print(f"[DeleteFieldAgent] seeded table  : {current_schema['table_name']}")
    print(f"[DeleteFieldAgent] seeded field  : {current_schema['field_name']}")

    system_prompt = (
        "You are a strict Database Schema Extraction Agent for an AI database assistant.\n\n"
        "Your job is to extract FIELD DELETION instructions from user input.\n"
        "You DO NOT generate conversational text. You ONLY return structured JSON.\n\n"
        "CRITICAL RULES:\n"
        "1. Extract the TARGET collection name (entity noun: 'product', 'customer', etc.)\n"
        "   NEVER use: 'collection','table','field','column','delete','remove' as names.\n"
        "2. Extract the TARGET field name to delete.\n"
        "   NEVER use: 'column','field','delete','remove','it' as field names.\n"
        "3. Use conversation history to resolve ambiguous references.\n"
        "   Example: if history shows 'add price to product' and user says 'delete it'\n"
        "   → table_name='product', field_name='price'\n"
        "4. If table_name or field_name cannot be identified → set to null.\n\n"
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
        print(f"[DeleteFieldAgent] LLM raw: {clean}")
        extracted = json.loads(clean)

        new_table_name = extracted.get("table_name")
        if new_table_name:
            maybe_reset_schema(state, new_table_name)
            current_schema["table_name"] = new_table_name

        if extracted.get("field_name"):
            current_schema["field_name"] = extracted["field_name"]

        state["schema_data"] = current_schema

        # Missing field detection
        llm_missing = extracted.get("missing_fields") or []
        missing = list(llm_missing)

        if not current_schema.get("table_name") and "table_name" not in missing:
            missing.append("table_name")
        if not current_schema.get("field_name") and "field_name" not in missing:
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
