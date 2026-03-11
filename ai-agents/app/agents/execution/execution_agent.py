from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.graph.state import AgentState
import requests
import json


# Operations that call the modify-schema bridge
MODIFY_SCHEMA_OPS = {"add_column", "update_collection", "update_field", "delete_field"}


async def execution_agent(state: AgentState) -> AgentState:
    """
    ExecutionAgent: upgraded to support batch execution and graceful duplicate skip.
    """
    print("\n----- ENTERING ExecutionAgent (Design Refined) -----")

    # 1. Resolve payloads
    execution_payloads = state.get("execution_payloads", [])
    strapi_payload     = state.get("strapi_payload")
    
    if execution_payloads:
        payloads = execution_payloads
    elif strapi_payload:
        payloads = [strapi_payload] if not isinstance(strapi_payload, list) else strapi_payload
    else:
        state["execution_error"] = "No Strapi payload available."
        return state

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    results = []
    errors  = []

    print(f"Executing {len(payloads)} payload(s).")

    for idx, payload in enumerate(payloads):
        override_endpoint = state.get("strapi_endpoint")
        operation        = payload.get("operation")

        if operation == "create_collection":
            url = "http://strapi:1337/api/ai-schema/create-collection"
        elif operation in MODIFY_SCHEMA_OPS:
            url = "http://strapi:1337/api/ai-schema/modify-schema"
        elif override_endpoint:
            url = override_endpoint
        else:
            url = "http://strapi:1337/api/ai-schema/create-collection"

        print(f"[{idx}] Operation : {operation}")
        print(f"[{idx}] Endpoint  : {url}")

        # ── LLM Gatekeeper ─────────────────────────────────────────────
        gatekeeper_prompt = (
            f"You are the final execution gatekeeper.\n"
            f"Payload:\n{json.dumps(payload, indent=2)}\n\n"
            "Validate payload for Strapi operation.\n"
            "Respond EXECUTE or BLOCK: <reason>"
        )

        check_response = await llm.ainvoke([SystemMessage(content=gatekeeper_prompt)])
        decision = check_response.content.strip()
        
        if not decision.startswith("EXECUTE"):
            print(f"[{idx}] decision → {decision}")
            errors.append(f"Payload {idx} blocked: {decision}")
            continue

        # ── Execution ──────────────────────────────────────────────────
        try:
            print(f"[{idx}] Requesting POST {url}")
            resp = requests.post(url, json=payload, timeout=30)
            print(f"[{idx}] Status: {resp.status_code}")
            
            if resp.status_code == 200:
                results.append(resp.json())
            elif resp.status_code == 400:
                # ── Problem 3 Fix: Graceful duplicate skip ────────────────
                error_body = resp.text.lower()
                if "already exists" in error_body:
                    msg = f"ExecutionAgent: Duplicate collection skipped: {payload.get('singularName') or payload.get('collection')}"
                    print(f"[{idx}] {msg}")
                    results.append({"status": "skipped", "reason": "already exists"})
                    continue
                else:
                    errors.append(f"Payload {idx} failed (HTTP 400): {resp.text}")
            else:
                errors.append(f"Payload {idx} failed (HTTP {resp.status_code}): {resp.text}")
        except Exception as e:
            errors.append(f"Payload {idx} runtime error: {str(e)}")

    # ── Final State Update ─────────────────────────────────────────────
    state["execution_result"] = results if len(payloads) > 1 else (results[0] if results else None)
    
    if errors:
        state["execution_error"]   = " | ".join(errors)
        state["interaction_phase"] = True
        state["active_agent"]      = "interaction_planner"
    else:
        state["execution_error"]   = None
        state["interaction_phase"] = False
        print("ExecutionAgent: Batch completed successfully.")

        # Cleanup
        state["execution_payloads"] = []
        state["strapi_payload"]     = {}
        state["strapi_endpoint"]    = ""
        state["operation"]          = ""
        state["schema_ready"]       = False

    return state
