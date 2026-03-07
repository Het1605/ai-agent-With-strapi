import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from app.graph.state import AgentState

load_dotenv()

def supervisor_agent(state: AgentState) -> AgentState:
    """
    SupervisorAgent: A simple agent that uses an LLM to respond to the user.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    user_message = state.get("user_input", "")
    
    # Call LLM
    ai_response = llm.invoke([HumanMessage(content=user_message)])
    
    # Update state
    state["response"] = ai_response.content
    return state
