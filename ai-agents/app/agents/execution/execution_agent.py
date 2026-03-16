import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.graph.state import AgentState
from app.services.strapi_client import strapi_client

import asyncio

async def execution_agent(state: AgentState) -> AgentState:
    """
    ExecutionAgent (Sequential Executor): Safe sequential executor with retries.
    Delegates all networking to the 'strapi_client' service.
    """
    print("\n----- ENTERING ExecutionAgent (Sequential) -----")

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
            "Allowed operations include: add_column, delete_column, update_column, update_collection.\n"
            "DO NOT block 'delete_column' operations unless fundamentally malformed.\n"
            "Respond EXECUTE or BLOCK: <reason>"
        )
        check_response = await llm.ainvoke([SystemMessage(content=gatekeeper_prompt)])
        if not check_response.content.strip().startswith("EXECUTE"):
            msg = f"Payload {idx} blocked: {check_response.content.strip()}"
            print(f"[ExecutionAgent] {msg}")
            errors.append(msg)
            continue

        print("check_response:",check_response)

        # ── 2. Request Execution via Service (with retries) ────────────
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            response = strapi_client.post_payload(payload)
            
            if response["status"] == "success":
                results.append({
                    "operation": payload.get("operation"),
                    "collection": payload.get("collection") or payload.get("collectionName"),
                    "status": "success",
                    "data": response.get("data")
                })
                success = True
                break
            elif response["status"] == "skipped":
                results.append({
                    "operation": payload.get("operation"),
                    "collection": payload.get("collection") or payload.get("collectionName"),
                    "status": "skipped",
                    "reason": "already_exists"
                })
                success = True
                break
            else:
                print(f"[ExecutionAgent] Attempt {attempt+1}/{max_retries} failed: {response['error']}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # Delay between retries
        
        if not success:
            errors.append(f"Operation failed after {max_retries} attempts: {response['error']}")

    # ── 3. Final State Update ──────────────────────────────────────────
    state["execution_result"] = {"results": results}
    
    if errors:
        state["execution_error"] = " | ".join(errors)
        state["interaction_phase"] = True
    else:
        state["execution_error"] = None
        state["interaction_phase"] = False
        print("[ExecutionAgent] Batch completed. All operations executed.")

    # Cleanup state fully to prevent accidental reuse
    state["execution_payloads"] = []
    state["modify_schema_plan"] = {}
    state["modify_operations"] = []
    state["modify_schema_design"] = {}
    state["approval_status"] = None
    state["interaction_message"] = None
    state["schema_ready"] = False

    print("modify_schema_plan:",state["modify_schema_plan"])
    print("modify_schema_design:", state["modify_schema_design"])

    return state
