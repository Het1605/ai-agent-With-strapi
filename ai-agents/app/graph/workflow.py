from langgraph.graph import StateGraph, END
from app.graph.state import AgentState
from app.agents.supervisor.supervisor_agent import supervisor_agent

def create_workflow():
    """
    Creates a simple LangGraph workflow:
    SupervisorAgent -> END
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_agent)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add edge to END
    workflow.add_edge("supervisor", END)
    
    # Compile
    return workflow.compile()
