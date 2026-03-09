from app.graph.state import AgentState

def initialize_conversation_memory(state: AgentState) -> AgentState:
    """
    Initializes the conversation history in the state if it doesn't exist.
    """
    if "conversation_history" not in state or state["conversation_history"] is None:
        state["conversation_history"] = []
    return state

def add_user_message(state: AgentState) -> AgentState:
    """
    Adds the current user query to the conversation history.
    """
    user_query = state.get("user_query") or state.get("user_input")
    if user_query:
        state["conversation_history"].append({
            "role": "user",
            "content": user_query
        })
    return trim_history(state)

def add_ai_message(state: AgentState, message: str) -> AgentState:
    """
    Adds the AI's response to the conversation history.
    """
    if message:
        state["conversation_history"].append({
            "role": "assistant",
            "content": message
        })
    return trim_history(state)

def trim_history(state: AgentState) -> AgentState:
    """
    Ensures the conversation history does not exceed 20 messages.
    """
    if len(state["conversation_history"]) > 20:
        state["conversation_history"] = state["conversation_history"][-20:]
    return state
