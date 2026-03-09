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
        "debug_info": state.get("debug_info"),
        "interaction_message": state.get("interaction_message"),
        "execution_result": state.get("execution_result"),
        "execution_error": state.get("execution_error"),
    }
    
    system_prompt = (
        "You are the Voice of the College Management System. "
        "Your role is to take the internal analysis and context from various agents "
        "and format a final, natural language response for the user.\n\n"
        "DATABASE REPORTING RULES:\n"
        "1. If 'execution_error' exists and is not empty, explain the failure friendly but clearly.\n"
        "2. If 'execution_result' exists, celebrate the success and summarize what was created.\n"
        "3. If 'interaction_message' exists, present it as the next step the user should take.\n"
        "4. Otherwise, summarize the planned task context: 'Detected Task: <planned_task>\\nUser Query: <user_input>'.\n\n"
        "Tone: Friendly, professional, and helpful."
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
