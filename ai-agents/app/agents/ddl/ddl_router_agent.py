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

            Your job is to classify a user request into ONE of:

            DDL_CREATE_TABLE
            DDL_MODIFY_SCHEMA

            --------------------------------------------------
            CRITICAL PROBLEM TO FIX
            --------------------------------------------------

            The system incorrectly routes requests that involve extending an existing system.

            Example:
            "add salary tables to existing employee system"

            This is NOT modification.
            This is EXTENSION → CREATE.

            --------------------------------------------------
            HOW TO THINK (MANDATORY)
            --------------------------------------------------

            You must classify the request into ONE of THREE intent types:

            1. NEW SYSTEM
            → User is starting from scratch

            2. EXTENDING SYSTEM (VERY IMPORTANT)
            → User is adding NEW tables/modules
            → May reference existing tables
            → BUT NOT modifying their structure

            3. MODIFYING SYSTEM
            → User is changing existing tables/columns

            --------------------------------------------------
            ROUTING DECISION
            --------------------------------------------------

            → If NEW SYSTEM → DDL_CREATE_TABLE  
            → If EXTENDING SYSTEM → DDL_CREATE_TABLE ✅  
            → If MODIFYING SYSTEM → DDL_MODIFY_SCHEMA  

            --------------------------------------------------
            KEY DIFFERENCE (CRITICAL)
            --------------------------------------------------

            EXTENSION vs MODIFICATION:

            EXTENSION:
            - "add tables"
            - "add module"
            - "add salary system"
            - "add feature"
            - "integrate new functionality"

            → CREATE ✅

            MODIFICATION:
            - "add column"
            - "remove field"
            - "update table"
            - "change schema"

            → MODIFY

            --------------------------------------------------
            CONTEXT RULE
            --------------------------------------------------

            Even if existing tables are mentioned:

            IF user is adding NEW tables → CREATE

            --------------------------------------------------
            AMBIGUITY HANDLING
            --------------------------------------------------

            If unsure:

            - If new entities/tables/modules are involved → CREATE
            - Only choose MODIFY if clearly editing existing structure

            --------------------------------------------------
            STRICT RULES
            --------------------------------------------------

            ❌ Do NOT rely on keywords like "existing"
            ❌ Do NOT confuse extension with modification

            ✅ Focus on what is being CREATED vs CHANGED

            --------------------------------------------------
            OUTPUT
            --------------------------------------------------

            Respond with EXACTLY one:

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
