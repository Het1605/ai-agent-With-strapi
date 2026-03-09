from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

async def interaction_planner_agent(state: AgentState) -> AgentState:
    """
    InteractionPlannerAgent: Manages the interaction loop. 
    It either asks for missing data OR processes the provided data and prepares to loop back.
    """
    print("\n----- ENTERING InteractionPlannerAgent -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    execution_error = state.get("execution_error")
    
    if execution_error:
        print(f"InteractionPlannerAgent: Handling Execution Error: {execution_error}")
        prompt = (
            "You are a Database Assistant. The user tried to execute a database operation, but the backend returned an error:\n"
            f"ERROR: {execution_error}\n\n"
            "Generate a friendly, helpful conversational response asking the user how they would like to resolve this issue."
        )
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        state["response"] = response.content.strip()
        state["interaction_phase"] = True
        state["active_agent"] = "create_table"
        state["execution_error"] = None
        state["interaction_attempts"] = state.get("interaction_attempts", 0) + 1
        state["user_provided_missing_data"] = False
        return state

    # Standard Schema Missing Information Flow
    missing_fields = state.get("missing_fields", [])
    schema_data = state.get("schema_data", {})
    
    state["interaction_attempts"] = state.get("interaction_attempts", 0) + 1
    
    if state["interaction_attempts"] > 5:
        state["response"] = "I'm having trouble understanding. Could you please provide the missing table details more clearly, or let's start over?"
        state["interaction_phase"] = True
        state["active_agent"] = "create_table"
        state["user_provided_missing_data"] = False
        return state

    request_prompt = (
        "You are a Database Assistant. A user wants to construct a table but is missing some information.\n"
        f"Missing Fields: {missing_fields}\n"
        f"Current Schema: {json.dumps(schema_data)}\n\n"
        "Generate a friendly, professional question to ask the user for this missing information."
    )
    
    request_response = await llm.ainvoke([SystemMessage(content=request_prompt)])
    state["response"] = request_response.content.strip()
    
    state["interaction_phase"] = True
    state["active_agent"] = "create_table"
    state["user_provided_missing_data"] = False
    
    print(f"InteractionPlannerAgent: Asking user -> {state['response']}")
    
    return state
