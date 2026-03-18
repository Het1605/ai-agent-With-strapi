from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def scope_classifier_agent(state: AgentState) -> AgentState:
    """
    ScopeClassifierAgent: Classifies queries into conversation, general, or database.
    """
    print("----- ENTERING ScopeClassifierAgent -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    user_input = state.get("user_input", "")
    history = state.get("conversation_history", [])
    
    # Format history for prompt
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
    
    system_prompt = f"""
            You are a highly accurate Scope Classification Agent.

            Your job is to classify the user’s request into EXACTLY one of these three scopes:

            1. conversation
            2. database
            3. general

            --------------------------------------------------
            🚨 CORE PROBLEM TO FIX
            --------------------------------------------------

            The system is incorrectly classifying database-related queries as "general".

            This MUST NOT happen.

            When in doubt → ALWAYS prefer "database" over "general".

            --------------------------------------------------
            🧠 CONTEXT (VERY IMPORTANT)
            --------------------------------------------------

            Recent Conversation History:
            {history_str}

            You MUST use this history to understand user intent.

            If the conversation is already about database/schema,
            then even vague follow-ups MUST be classified as "database".

            --------------------------------------------------
            🧠 CLASSIFICATION RULES
            --------------------------------------------------

            1. conversation:
            - Greetings, casual talk, identity questions
            - Examples:
            "hi", "hello", "who are you", "how are you"

            --------------------------------------------------

            2. database (VERY IMPORTANT - HIGH PRIORITY):
            Classify as "database" if the user is:

            ✔ Asking to design, create, or build a system/schema  
            ✔ Talking about storing data  
            ✔ Describing what kind of data they want to manage  
            ✔ Referring to entities, records, fields, or structure  
            ✔ Asking to modify or extend an existing system  
            ✔ Responding to previous schema-related questions  
            ✔ Using indirect language like:
            - "I want to store..."
            - "I need a system for..."
            - "track/manage data..."
            - "handle records..."

            IMPORTANT:
            Even if NO table/column names are mentioned → STILL database

            Examples:
            - "I want to store employee monthly data"
            - "build system for managing orders"
            - "track user activity"
            - "design schema for hospital"
            - "add salary feature"
            - "manage bookings and payments"

            → ALL of these are DATABASE

            --------------------------------------------------

            3. general:
            - Only pure knowledge questions unrelated to any system design or data storage

            Examples:
            - "what is AI?"
            - "who is prime minister of India"
            - "explain cloud computing"

            --------------------------------------------------
            ⚠️ PRIORITY ORDER (CRITICAL)
            --------------------------------------------------

            If a query can belong to BOTH:

            database vs general → choose DATABASE ✅

            conversation vs database → choose DATABASE (if any system/data hint exists)

            --------------------------------------------------
            🚫 STRICT RULES
            --------------------------------------------------

            ❌ Do NOT overuse "general"
            ❌ Do NOT ignore data-related intent
            ❌ Do NOT require explicit table names

            ✅ Be biased toward DATABASE classification

            --------------------------------------------------
            OUTPUT
            --------------------------------------------------

            Respond ONLY with one word:

            conversation
            database
            general
        """
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])
    
    scope = response.content.strip().lower()
    if scope not in ["conversation", "general", "database"]:
        scope = "general"
        
    state["scope"] = scope
    print(f"Detected scope: {scope}")
    
    state["analysis"] = (state.get("analysis") or "") + f"\nClassification: Resolved as '{scope}' scope."
        
    return state
