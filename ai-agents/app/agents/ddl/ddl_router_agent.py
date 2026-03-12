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
      DDL_CREATE_TABLE  — design new collections/database (routes to RequirementAgent pipeline)
      DDL_MODIFY_SCHEMA — estrutural changes to existing collections (routes to ModifySchemaAgent)
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
        "You are an expert Routing Agent for Database Schema Operations.\n"
        "Your task is to classify user requests into exactly one of two supported DDL operations.\n\n"
        "SUPPORTED OPERATIONS:\n"
        "1. DDL_CREATE_TABLE: Use this for brand new database or table design tasks.\n"
        "   - This routes to the RequirementAgent → PlanningAgent → SchemaDesignerAgent pipeline.\n"
        "   - Use this for any 'design', 'build', or 'new' creation requests.\n"
        "   - Examples: 'create employee table', 'create new collection order', 'design ecommerce database', 'build database for school', 'create full DB for hospital', 'design database architecture for restaurant system'.\n\n"
        "2. DDL_MODIFY_SCHEMA: Use this for any structural changes to an existing schema.\n"
        "   - This routes to the ModifySchemaAgent.\n"
        "   - Examples: 'add column', 'remove field', 'rename column', 'change constraints', 'set default', 'delete collection', 'update field settings'.\n\n"
        "ROUTING RULES:\n"
        "- If the request is for a brand new design or a full database architecture (multiple tables) → DDL_CREATE_TABLE.\n"
        "- If the request references an existing table/column or seeks to alter an existing structure → DDL_MODIFY_SCHEMA.\n"
        "- If the result is DDL_CREATE_TABLE, the workflow will route to RequirementAgent for analysis.\n"
        "- If the result is DDL_MODIFY_SCHEMA, the workflow will route to ModifySchemaAgent for processing.\n\n"
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
