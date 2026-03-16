from app.graph.state import AgentState
from langgraph.types import interrupt

async def user_approval_agent(state: AgentState) -> AgentState:
    """
    UserApprovalAgent (Modify Schema Flow): Pure Human-in-the-Loop Controller.
    Triggers interrupt() using the already-set state["response"].
    """
    print("\n----- ENTERING ModifySchemaUserApprovalAgent -----")
    
    # The message should already be in state["response"] from either:
    # 1. ModifySchemaVisualizationAgent (Initial pass)
    # 2. ModifySchemaUserRepromptAgent (Invalid input pass)
    display_message = state.get("response", "No schema modification preview available. Do you want to proceed?")

    print(f"[ModifySchemaUserApprovalAgent] Triggering interrupt with message: {display_message[:50]}...")
    
    # 3. Human-in-the-Loop Interrupt
    user_response = interrupt({
        "type": "schema_modification_approval",
        "message": display_message
    })

    print(f"[ModifySchemaUserApprovalAgent] Resumed. Captured user feedback: {user_response}")
    
    # 4. Capture feedback for routing
    state["user_input"] = user_response
    state["interaction_phase"] = True
    
    return state
