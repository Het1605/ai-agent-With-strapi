from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def conversation_agent(state: AgentState) -> AgentState:
    """
    ConversationAgent: Provides friendly, conversational analysis.
    """
    print("----- ENTERING ConversationAgent -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    user_input = state.get("user_input", "")
    
    system_prompt = (
        "You are a friendly and helpful assistant for a College Management System. "
        "The user is engaging in casual conversation. "
        "Analyze the user's intent and provide a friendly, helpful context for the final response. "
        "Ignore minor typos in greetings (e.g., 'hellow' is a friendly hello). "
        "Do NOT speak to the user. Speak ONLY to the system by providing internal analysis. "
        "Ensure the final persona suggested is warm, casual, and helpful."
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])
    
    state["analysis"] = response.content
    state["intent"] = "casual_conversation"
    
    return state
