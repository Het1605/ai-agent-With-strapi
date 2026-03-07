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
    
    system_prompt = (
        "Classify the user's request into exactly one of three scopes:\n"
        "1. 'conversation': For greetings, small talk, or self-introductions (e.g., 'hi', 'who are you').\n"
        "2. 'database': For requests to query, create, modify, or delete collections or records in a database.\n"
        "3. 'general': For general knowledge questions not related to the database.\n\n"
        "Respond ONLY with the category name: conversation, general, or database."
    )
    
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
