from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()

async def supervisor_agent(state: AgentState) -> AgentState:
    """
    SupervisorAgent: Orchestrates the entry point of the multi-agent workflow.
    """
    print("\n----- ENTERING SupervisorAgent -----")
    
    # Return state to pass to validation
    return state
