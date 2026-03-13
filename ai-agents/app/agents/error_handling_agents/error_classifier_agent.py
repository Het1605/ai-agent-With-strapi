from app.graph.state import AgentState

async def error_classifier_agent(state: AgentState) -> AgentState:
    """
    ErrorClassifierAgent: Categorizes the root cause based on parallel analysis outputs.
    """
    print("\n----- ENTERING ErrorClassifierAgent -----")
    return state
