from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

# The only two DDL operations the system supports
_VALID_DDL_OPS = {"DDL_CREATE_TABLE", "DDL_MODIFY_SCHEMA"}


async def ddl_router_agent(state: AgentState) -> AgentState:
    """
    DDLRouterAgent: determines the exact DDL operation type.

    Only two operations are supported:
      DDL_CREATE_TABLE  — create a new collection from scratch
      DDL_MODIFY_SCHEMA — any structural change to an existing collection
                          (add/remove/rename fields, delete collection, etc.)

    Any unsupported type (e.g. DDL_DELETE_TABLE) is silently normalised to
    DDL_MODIFY_SCHEMA.
    """
    print("\n----- ENTERING DDLRouterAgent -----")

    if state.get("interaction_phase") == True:
        print("DDLRouterAgent: Bypassing during interaction phase.")
        return state

    task_queue    = state.get("task_queue", [])
    current_index = state.get("current_task_index", 0)
    current_task  = task_queue[current_index] if current_index < len(task_queue) else {}

    task_type = current_task.get("task_type", "")

    # ── Deterministic fast-path: task_type already set ──
    if task_type in _VALID_DDL_OPS:
        state["ddl_operation"] = task_type
        print(f"DDLRouterAgent: Deterministic route → {task_type}")
        state["analysis"] = (state.get("analysis") or "") + f"\nDDL Router: Resolved operation as {task_type} (deterministic)."
        return state

    # ── Safety normalisation: rewrite unsupported to DDL_MODIFY_SCHEMA ──────
    if task_type == "DDL_DELETE_TABLE":
        print(f"DDLRouterAgent: DDL_DELETE_TABLE is not supported → normalising to DDL_MODIFY_SCHEMA")
        state["ddl_operation"] = "DDL_MODIFY_SCHEMA"
        state["analysis"] = (state.get("analysis") or "") + "\nDDL Router: DDL_DELETE_TABLE normalised to DDL_MODIFY_SCHEMA."
        return state

    # ── LLM fallback for ambiguous or unknown task types ────────────────────
    print(f"DDLRouterAgent: Unrecognised task_type '{task_type}' — calling LLM fallback.")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = (
        "You are a routing agent for database schema operations.\n\n"
        "The system supports ONLY two DDL operations:\n"
        "  DDL_CREATE_TABLE\n"
        "  DDL_MODIFY_SCHEMA\n\n"
        "Rules:\n"
        "1. Creating a new table/collection from scratch → DDL_CREATE_TABLE\n"
        "2. Any structural modification of an existing schema (add/remove/rename fields,\n"
        "   delete collection, rename collection, change constraints, set defaults) → DDL_MODIFY_SCHEMA\n"
        "3. DDL_DELETE_TABLE does NOT exist. Collection deletion = DDL_MODIFY_SCHEMA.\n\n"
        "Respond ONLY with exactly one of:\n"
        "DDL_CREATE_TABLE\n"
        "DDL_MODIFY_SCHEMA"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Task Context: {current_task}")
    ])

    operation = response.content.strip()

    # Validate LLM output — reject anything outside the allowed set
    if operation not in _VALID_DDL_OPS:
        print(f"DDLRouterAgent: LLM returned unknown op '{operation}' — defaulting to DDL_MODIFY_SCHEMA.")
        operation = "DDL_MODIFY_SCHEMA"

    state["ddl_operation"] = operation
    print(f"DDLRouterAgent: Determined operation → {operation}")
    state["analysis"] = (state.get("analysis") or "") + f"\nDDL Router: Resolved operation as {operation} (LLM fallback)."

    return state
