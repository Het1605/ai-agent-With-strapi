from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

# The only two DDL operations the system supports
_VALID_DDL_OPS = {"DDL_CREATE_TABLE", "DDL_MODIFY_SCHEMA"}


async def ddl_router_agent(state: AgentState) -> AgentState:
    """
    DDLRouterAgent: determines the exact DDL operation type.
    Now upgraded to be context-aware (user_input + history).

    Supported operations:
      DDL_CREATE_TABLE  — create a new collection / full database design
      DDL_MODIFY_SCHEMA — any structural change to an existing collection
    """
    print("\n----- ENTERING DDLRouterAgent (Context Aware) -----")

    if state.get("interaction_phase") == True:
        print("DDLRouterAgent: Bypassing during interaction phase.")
        return state

    user_input = state.get("user_input", "")
    history    = state.get("conversation_history", [])[-5:]
    
    print(f"[DDLRouter] User Input: {user_input}")
    
    # ── LLM classification with full context ────────────────────────────
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = (
        "You are a routing agent for database schema operations.\n\n"
        "The system supports ONLY two DDL operations:\n"
        "  DDL_CREATE_TABLE\n"
        "  DDL_MODIFY_SCHEMA\n\n"
        "Routing Rules:\n"
        "1. If the user is creating a brand new table or collection → DDL_CREATE_TABLE\n"
        "   Examples: 'create employee table', 'build database for school', 'design ecommerce'.\n"
        "2. If the user modifies any existing schema structure → DDL_MODIFY_SCHEMA\n"
        "   Examples: 'add column', 'remove field', 'rename column', 'change constraints', 'set default', 'delete collection'.\n"
        "3. Table deletion must be routed as DDL_MODIFY_SCHEMA.\n"
        "4. If the request references an existing table or column, it is always DDL_MODIFY_SCHEMA.\n"
        "5. If the user requests a full database design (multiple tables), it is DDL_CREATE_TABLE.\n\n"
        "Respond ONLY with exactly one of:\n"
        "DDL_CREATE_TABLE\n"
        "DDL_MODIFY_SCHEMA"
    )

    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    human_msg = (
        f"Conversation History:\n{history_str}\n\n"
        f"Current User Request: {user_input}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    operation = response.content.strip().upper()

    # Validate LLM output
    if operation not in _VALID_DDL_OPS:
        print(f"[DDLRouter] LLM returned unknown op '{operation}' — defaulting to DDL_MODIFY_SCHEMA.")
        operation = "DDL_MODIFY_SCHEMA"

    state["ddl_operation"] = operation
    print(f"[DDLRouter] Decision: {operation}")
    state["analysis"] = (state.get("analysis") or "") + f"\nDDL Router: Resolved as {operation} (context-aware)."

    return state
