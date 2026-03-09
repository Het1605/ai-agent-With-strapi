from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import requests
import json

async def execution_agent(state: AgentState) -> AgentState:
    """
    ExecutionAgent: AI-driven agent that executes the database operation via Strapi Bridge.
    It uses an LLM to 'authorize' the final request and then analyze the response.
    """
    print("\n----- ENTERING ExecutionAgent -----")
    
    payload = state.get("strapi_payload")
    if not payload:
        state["execution_error"] = "No Strapi payload available for execution."
        return state

    # User requested ExecutionAgent has an LLM agent
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    confirm_prompt = (
        f"You are the Final Execution Gatekeeper. Analyze this Strapi payload:\n{json.dumps(payload, indent=2)}\n\n"
        "Confirm if this payload is valid for a 'create-collection' operation. "
        "If yes, respond with 'EXECUTE'. If there are fatal errors, respond with 'BLOCK: <reason>'."
    )
    
    check_response = await llm.ainvoke([SystemMessage(content=confirm_prompt)])
    decision = check_response.content.strip()
    
    if not decision.startswith("EXECUTE"):
        state["execution_error"] = f"Execution blocked by AI: {decision}"
        return state

    # Programmatic Execution
    url = "http://strapi:1337/api/ai-schema/create-collection"
    try:
        print(f"ExecutionAgent: Sending POST request to {url}")
        resp = requests.post(url, json=payload, timeout=30)
        
        if resp.status_code == 200:
            result = resp.json()
            state["execution_result"] = result
            state["execution_error"] = None
            state["interaction_phase"] = False
            print("ExecutionAgent: Successfully created collection.")
        else:
            error_data = resp.text
            state["execution_error"] = f"Strapi Error ({resp.status_code}): {error_data}"
            state["interaction_phase"] = True
            state["active_agent"] = "interaction_planner"
            print(f"ExecutionAgent: Failed with status {resp.status_code}")
            
    except Exception as e:
        print(f"ExecutionAgent: Runtime Error: {e}")
        state["execution_error"] = f"Network or Runtime Error: {str(e)}"
        state["interaction_phase"] = True
        state["active_agent"] = "interaction_planner"
    
    return state
