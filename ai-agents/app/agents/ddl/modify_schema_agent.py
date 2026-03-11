from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

VALID_OPERATIONS = {"add_column", "update_collection", "update_field", "delete_field"}

_ALLOWLIST = {
    "add_column":        ["table_name", "field_name", "field_type"],
    "update_collection": ["table_name", "new_display_name", "delete"],
    "update_field":      ["table_name", "field_name", "updates"],
    "delete_field":      ["table_name", "field_name"],
}

async def modify_schema_agent(state: AgentState) -> AgentState:
    """
    ModifySchemaAgent — LLM-based operation classifier + target metadata extractor.

    Now extracts:
      state["operation"]       — which sub-operation to run
      state["active_agent"]    — same value (for interaction resume)
      state["modify_operation"] — rich target context for downstream agents
    """
    print("\n----- ENTERING ModifySchemaAgent -----")

    user_input = state.get("user_input", "")
    print(f"User Input: {user_input}")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = (
        "You are a Schema Modification Classifier in a multi-agent database system.\n\n"
        "Your job is to classify the user's intent AND extract the target entities.\n\n"
        "SUPPORTED OPERATIONS:\n"
        "  add_column        — add new field(s) to an existing collection\n"
        "  update_collection — rename display name OR delete an entire collection\n"
        "  update_field      — change settings of an existing field (required, unique, rename, default, etc.)\n"
        "  delete_field      — remove a specific field from a collection\n\n"
        "CLASSIFICATION RULES:\n"
        "- 'add price to product', 'add column stock', 'add field email'            → add_column\n"
        "- 'rename collection', 'delete collection', 'delete table'                 → update_collection\n"
        "- 'rename column/field', 'make unique', 'set required', 'set default'      → update_field\n"
        "- 'delete column', 'remove field', 'drop field'                            → delete_field\n\n"
        "EXTRACTION RULES:\n"
        "- table_name: the entity noun ('product', 'student', 'order'). NEVER use 'collection','table','column'.\n"
        "- field_name: the specific field being targeted (for update_field / delete_field / add_column).\n"
        "- field_type: the type of new field (for add_column only, if mentioned).\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"operation": "<op>", "table_name": "<str|null>", "field_name": "<str|null>", "field_type": "<str|null>"}\n'
        "No explanation. No extra text."
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User request: {user_input}")
    ])

    operation    = "add_column"  # safe default
    target_table = None
    target_field = None
    field_type   = None

    try:
        clean  = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)

        raw_op = result.get("operation", "").strip()
        if raw_op in VALID_OPERATIONS:
            operation = raw_op
        else:
            print(f"[ModifySchemaAgent] Unknown operation '{raw_op}' — defaulting to 'add_column'")

        target_table = result.get("table_name") or None
        target_field = result.get("field_name") or None
        field_type   = result.get("field_type") or None

    except Exception as e:
        print(f"[ModifySchemaAgent] JSON parse error: {e} — defaulting to 'add_column'")

    # Write routing state
    state["operation"]    = operation
    state["active_agent"] = operation

    # Write structured target context for downstream agents
    state["modify_operation"] = {
        "operation":    operation,
        "target_table": target_table,
        "target_field": target_field,
        "field_type":   field_type,
    }

    print(f"Detected Operation : {operation}")
    print(f"Target Table       : {target_table}")
    print(f"Target Field       : {target_field}")
    print(f"Routing to agent   : {operation}")

    return state
