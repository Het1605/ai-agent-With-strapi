from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

VALID_OPERATIONS = {"add_column", "update_collection", "update_field", "delete_field"}

async def modify_schema_agent(state: AgentState) -> AgentState:
    """
    ModifySchemaAgent — LLM-based operation classifier.

    Reads the user's natural language request and determines which schema
    modification operation is intended. It does NOT extract schema details,
    does NOT call query_builder, and does NOT modify schema_data.

    Output:
        state["operation"]   — one of the VALID_OPERATIONS
        state["active_agent"] — same value (used by StateRouterAgent next turn)
    """
    print("\n----- ENTERING ModifySchemaAgent -----")

    user_input = state.get("user_input", "")
    print(f"User Input: {user_input}")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = (
        "You are a Schema Modification Classifier in a multi-agent database system.\n\n"
        "Your ONLY job is to classify the user's intent into ONE of these operations:\n\n"
        "  add_column        — user wants to add new field(s) to an existing collection\n"
        "  update_collection — user wants to rename, update settings, or DELETE an entire collection\n"
        "  update_field      — user wants to change settings of an existing field (required, unique, rename, etc.)\n"
        "  delete_field      — user wants to remove a specific field from a collection\n\n"
        "Classification rules:\n"
        "- 'add price to product', 'add column stock', 'add field email' → add_column\n"
        "- 'rename collection', 'delete collection', 'delete table', 'drop collection' → update_collection\n"
        "- 'make unique', 'set required', 'rename field', 'update field settings' → update_field\n"
        "- 'delete column', 'remove field', 'drop field' → delete_field\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"operation": "<chosen_operation>"}\n'
        "No explanation. No extra text."
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User request: {user_input}")
    ])

    # Parse and validate
    operation = "add_column"  # safe default
    try:
        clean = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        raw_op = result.get("operation", "").strip()
        if raw_op in VALID_OPERATIONS:
            operation = raw_op
        else:
            print(f"[ModifySchemaAgent] Unknown operation '{raw_op}' — defaulting to 'add_column'")
    except Exception as e:
        print(f"[ModifySchemaAgent] JSON parse error: {e} — defaulting to 'add_column'")

    state["operation"]    = operation
    state["active_agent"] = operation

    print(f"Detected Operation: {operation}")
    print(f"Routing to agent:   {operation}")

    return state
