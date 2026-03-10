from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.graph.state import AgentState
import requests
import json


# Operations that call the modify-schema bridge
MODIFY_SCHEMA_OPS = {"add_column", "update_collection", "update_field", "delete_field"}


async def execution_agent(state: AgentState) -> AgentState:
    """
    ExecutionAgent: operation-aware dynamic execution engine.

    Reads state["strapi_payload"] and state["operation"] to select the
    correct Strapi bridge endpoint, validates the payload with an LLM
    gatekeeper, then executes the request.
    """
    print("\n----- ENTERING ExecutionAgent -----")

    payload   = state.get("strapi_payload")
    operation = state.get("operation") or "create_collection"

    if not payload:
        state["execution_error"] = "No Strapi payload available for execution."
        return state

    # ── Endpoint selection ─────────────────────────────────────────────
    # QueryBuilderAgent may have already stored the correct URL; use it as
    # an override. Otherwise derive from operation type.
    if state.get("strapi_endpoint"):
        url = state["strapi_endpoint"]
    elif operation in MODIFY_SCHEMA_OPS:
        url = "http://strapi:1337/api/ai-schema/modify-schema"
    else:
        url = "http://strapi:1337/api/ai-schema/create-collection"

    print(f"Operation : {operation}")
    print(f"Endpoint  : {url}")

    # ── LLM Gatekeeper ────────────────────────────────────────────────
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    gatekeeper_prompt = (
        f"You are the final execution gatekeeper.\n\n"
        f"Operation: {operation}\n\n"
        f"Payload:\n{json.dumps(payload, indent=2)}\n\n"
        "Validate whether this payload is structurally correct for the operation.\n\n"
        "Rules:\n"
        "- create_collection → 'collectionName' required; 'fields' may be empty or absent.\n"
        "- add_column        → 'collection' and 'data.fields' required.\n"
        "- update_collection → 'collection' required; 'data' may contain delete=true or displayName.\n"
        "- update_field      → 'collection', 'data.field', and 'data.updates' required.\n"
        "- delete_field      → 'collection' and 'data.field' required.\n\n"
        "If the payload is valid respond exactly: EXECUTE\n"
        "If invalid respond exactly: BLOCK: <reason>"
    )

    check_response = await llm.ainvoke([SystemMessage(content=gatekeeper_prompt)])
    decision = check_response.content.strip()
    print(f"ExecutionAgent: Gatekeeper decision → {decision}")

    if not decision.startswith("EXECUTE"):
        state["execution_error"] = f"Execution blocked by AI gatekeeper: {decision}"
        return state

    # ── Execute request ────────────────────────────────────────────────
    try:
        print(f"ExecutionAgent: Request payload → {json.dumps(payload, indent=2)}")
        resp = requests.post(url, json=payload, timeout=30)
        print(f"ExecutionAgent: Response status  → {resp.status_code}")
        print(f"ExecutionAgent: Response body    → {resp.text}")

        if resp.status_code == 200:
            state["execution_result"]  = resp.json()
            state["execution_error"]   = None
            state["interaction_phase"] = False
            print("ExecutionAgent: Operation completed successfully.")
        else:
            state["execution_error"]   = f"Strapi returned HTTP {resp.status_code}: {resp.text}"
            state["execution_result"]  = None
            state["interaction_phase"] = True
            state["active_agent"]      = "interaction_planner"
            print(f"ExecutionAgent: Request failed — {resp.status_code}")

    except Exception as e:
        print(f"ExecutionAgent: Runtime error → {e}")
        state["execution_error"]   = f"Network or runtime error: {str(e)}"
        state["execution_result"]  = None
        state["interaction_phase"] = True
        state["active_agent"]      = "interaction_planner"

    return state
