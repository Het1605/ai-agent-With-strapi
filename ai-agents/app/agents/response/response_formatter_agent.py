from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def response_formatter_agent(state: AgentState) -> AgentState:
    """
    ResponseFormatterAgent: The final "voice" of the system.
    """
    print("----- ENTERING ResponseFormatterAgent -----")
    print("Generating final response")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    context = {
        "user_input": state.get("user_input"),
        "scope": state.get("scope"),
        "intent": state.get("intent"),
        "analysis": state.get("analysis"),
        "validation_results": state.get("validation_results"),
    }
    
    system_prompt = (
        "You are the Voice of the College Management System. "
        "Your role is to take the internal analysis and context from various agents "
        "and format a final, natural language response for the user. "
        "Tone: Friendly, professional, and helpful. "
        "CRITICAL: If the analysis suggests a greeting or small talk, respond naturally "
        "and ignore minor typos. Do NOT lecture the user on spelling or grammar. "
        "Example: If user says 'hellow', just say 'Hey there! How can I help you today?'"
    )
    
    human_msg = (
        f"Context: {json.dumps(context, indent=2)}\n\n"
        "Provide a natural response to the user."
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])
    
    state["response"] = response.content
    return state
