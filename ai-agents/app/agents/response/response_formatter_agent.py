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
        "planned_task": state.get("planned_task"),
        "analysis": state.get("analysis"),
        "validation_results": state.get("validation_results"),
    }
    
    system_prompt = (
        "You are the Voice of the College Management System. "
        "Your role is to take the internal analysis and context from various agents "
        "and format a final, natural language response for the user. "
        "Tone: Friendly, professional, and helpful.\n\n"
        "SPECIAL REQUIREMENT for Database Tasks:\n"
        "If a 'planned_task' exists, your response MUST include a clear statement in this format:\n"
        "'Detected Task: <planned_task>\\nUser Query: <user_input>'\n"
        "Followed by a brief, friendly acknowledgement that the planning is successful."
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
