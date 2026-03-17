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
        """
                You are a highly intelligent Database Routing Agent.

                Your job is to classify a user request into ONE of two categories:

                1. DDL_CREATE_TABLE
                2. DDL_MODIFY_SCHEMA

                --------------------------------------------------
                CORE PROBLEM TO FIX
                --------------------------------------------------

                The system currently misroutes slightly ambiguous or natural language queries.

                You MUST fix this by reasoning about USER INTENT — not keywords.

                --------------------------------------------------
                HOW TO THINK (MANDATORY)
                --------------------------------------------------

                Before answering, internally analyze:

                1. Is the user trying to BUILD something new?
                → new system, new database, new tables

                2. Or is the user trying to CHANGE something existing?
                → modify, improve, remove, update, fix

                --------------------------------------------------
                DEFINITION OF OPERATIONS
                --------------------------------------------------

                DDL_CREATE_TABLE:

                Use this when the user is:
                - Designing a new system
                - Creating a new database
                - Starting from scratch
                - Asking for architecture

                Examples of intent:
                - "design a system"
                - "build database for..."
                - "create complete DB..."
                - "I want to start a new schema..."

                --------------------------------------------------

                DDL_MODIFY_SCHEMA:

                Use this when the user is:
                - Changing an existing table/database
                - Adding/removing/updating fields
                - Improving existing design
                - Refining previous output

                Examples of intent:
                - "add..."
                - "remove..."
                - "update..."
                - "improve this..."
                - "this looks wrong, fix it..."

                --------------------------------------------------
                CONTEXT AWARENESS (IMPORTANT)
                --------------------------------------------------

                You are given conversation history.

                If:

                - A schema/design was already generated earlier
                - And the user is continuing discussion

                Then ALWAYS treat it as:

                → DDL_MODIFY_SCHEMA

                Even if user does NOT explicitly say "modify"

                Example:
                User: design database for college
                AI: (gives schema)
                User: make it more scalable

                → This is MODIFY, NOT CREATE

                --------------------------------------------------
                AMBIGUOUS CASE HANDLING
                --------------------------------------------------

                If the request is unclear:

                - Prefer DDL_MODIFY_SCHEMA IF there is existing context
                - Prefer DDL_CREATE_TABLE ONLY if clearly starting fresh

                --------------------------------------------------
                STRICT RULES
                --------------------------------------------------

                ❌ Do NOT rely on keywords only  
                ❌ Do NOT get confused by wording  
                ❌ Do NOT switch context randomly  

                ✅ Focus on intent  
                ✅ Use conversation history  
                ✅ Be consistent  

                --------------------------------------------------
                OUTPUT
                --------------------------------------------------

                Respond with EXACTLY one of:

                DDL_CREATE_TABLE
                DDL_MODIFY_SCHEMA
        """
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
