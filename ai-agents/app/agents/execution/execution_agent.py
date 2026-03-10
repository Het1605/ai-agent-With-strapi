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
        "Confirm if this payload is valid for a Strapi bridge operation.\n"
        "IMPORTANT RULES:\n"
        "- An empty 'fields' array is VALID. Strapi allows collections with zero fields.\n"
        "- For create-collection: block only if 'collectionName' is missing or empty.\n"
        "- For modify-schema: block only if 'collectionName' or 'fields' is missing.\n"
        "- Do NOT block on empty fields array.\n"
        "If the payload is valid, respond with exactly 'EXECUTE'. "
        "If there is a truly fatal structural error, respond with 'BLOCK: <reason>'."
    )
    
    check_response = await llm.ainvoke([SystemMessage(content=confirm_prompt)])
    decision = check_response.content.strip()
    print(f"ExecutionAgent: Gatekeeper decision → {decision}")
    
    if not decision.startswith("EXECUTE"):
        state["execution_error"] = f"Execution blocked by AI: {decision}"
        return state

    # Select endpoint based on operation type (set by QueryBuilderAgent)
    url = state.get("strapi_endpoint") or "http://strapi:1337/api/ai-schema/create-collection"
    try:
        print(f"ExecutionAgent: Sending POST to {url}")
        print(f"ExecutionAgent: Request payload → {json.dumps(payload, indent=2)}")
        resp = requests.post(url, json=payload, timeout=30)
        print(f"ExecutionAgent: Response status  → {resp.status_code}")
        print(f"ExecutionAgent: Response body    → {resp.text}")
        
        if resp.status_code == 200:
            result = resp.json()
            state["execution_result"] = result
            state["execution_error"] = None
            state["interaction_phase"] = False
            print("ExecutionAgent: Collection created successfully.")
        else:
            # Store both status and the raw body so ResponseFormatterAgent can relay it accurately
            error_body = resp.text
            state["execution_error"] = f"Strapi returned HTTP {resp.status_code}: {error_body}"
            state["execution_result"] = None
            state["interaction_phase"] = True
            state["active_agent"] = "interaction_planner"
            print(f"ExecutionAgent: Request failed — status={resp.status_code}, body={error_body}")
            
    except Exception as e:
        print(f"ExecutionAgent: Runtime error → {e}")
        state["execution_error"] = f"Network or Runtime Error: {str(e)}"
        state["execution_result"] = None
        state["interaction_phase"] = True
        state["active_agent"] = "interaction_planner"
    
    return state
