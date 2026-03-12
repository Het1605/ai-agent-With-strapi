import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.graph.state import AgentState
from app.services.strapi_client import strapi_client

async def execution_agent(state: AgentState) -> AgentState:
    """
    ExecutionAgent (Lean): Pure Gatekeeper and State Manager.
    Delegates all networking and retries to the 'strapi_client' service.
    """
    print("\n----- ENTERING ExecutionAgent (Lean & Secure) -----")

    payloads = state.get("execution_payloads", [])
    if not payloads:
        state["execution_error"] = "No payloads to execute."
        return state

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    results = []
    errors = []

    for idx, payload in enumerate(payloads):
        print(f"[ExecutionAgent] Processing payload {idx+1}/{len(payloads)}")

        # ── 1. LLM Gatekeeper ──────────────────────────────────────────
        gatekeeper_prompt = (
            f"You are the final execution gatekeeper.\n"
            f"Payload:\n{json.dumps(payload, indent=2)}\n\n"
            "Validate payload for Strapi operation.\n"
            "Respond EXECUTE or BLOCK: <reason>"
        )
        check_response = await llm.ainvoke([SystemMessage(content=gatekeeper_prompt)])
        if not check_response.content.strip().startswith("EXECUTE"):
            msg = f"Payload {idx} blocked: {check_response.content.strip()}"
            print(f"[ExecutionAgent] {msg}")
            errors.append(msg)
            continue

        # ── 2. Request Execution via Service ──────────────────────────
        response = strapi_client.post_payload(payload)
        
        if response["status"] == "success":
            results.append(response["data"])
        elif response["status"] == "skipped":
            results.append({"status": "skipped", "reason": "already_exists"})
        else:
            errors.append(response["error"])

    # ── 3. Final State Update ──────────────────────────────────────────
    state["execution_result"] = results if len(results) > 1 else (results[0] if results else None)
    
    if errors:
        state["execution_error"] = " | ".join(errors)
        state["interaction_phase"] = True
    else:
        state["execution_error"] = None
        state["interaction_phase"] = False
        print("[ExecutionAgent] Batch completed successfully.")

    # Cleanup state
    state["execution_payloads"] = []
    state["schema_ready"] = False

    return state
