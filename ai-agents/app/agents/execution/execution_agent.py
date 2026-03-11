from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.graph.state import AgentState
import requests
import json
import time


# Operations that call the modify-schema bridge
MODIFY_SCHEMA_OPS = {"add_column", "update_collection", "update_field", "delete_field"}


async def execution_agent(state: AgentState) -> AgentState:
    """
    ExecutionAgent: upgraded to support batch execution, graceful duplicate skip,
    stabilization delay, and retry logic.
    """
    print("\n----- ENTERING ExecutionAgent (Reliability Refined) -----")

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

    total_payloads = len(payloads)
    print(f"Executing {total_payloads} payload(s).")

    for idx, payload in enumerate(payloads):
        current_num = idx + 1
        print(f"\n[ExecutionAgent] Executing payload {current_num}/{total_payloads}")

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

        # ── Execution with Retry Loop ──────────────────────────────────
        max_attempts = 3
        attempt = 0
        success = False

        while attempt < max_attempts:
            attempt += 1
            try:
                print(f"[{idx}] Attempt {attempt}/{max_attempts}: Requesting POST {url}")
                resp = requests.post(url, json=payload, timeout=30)
                
                print(f"[{idx}] Status: {resp.status_code}")
                try:
                    # Log raw response body for all responses
                    print(f"[{idx}] Response Body: {resp.text}")
                except Exception:
                    print(f"[{idx}] Unable to read response body")

                if resp.status_code == 200:
                    results.append(resp.json())
                    success = True
                    break
                
                # ── Error Handling ─────────────────────────────────────
                if resp.status_code >= 400:
                    try:
                        error_json = resp.json()
                        print(f"[{idx}] Parsed Error Detail: {json.dumps(error_json, indent=2)}")
                    except:
                        pass # Body already printed above as resp.text

                    if resp.status_code == 400:
                        error_body = resp.text.lower()
                        if "already exists" in error_body:
                            msg = f"ExecutionAgent: Duplicate collection skipped: {payload.get('singularName') or payload.get('collection')}"
                            print(f"[{idx}] {msg}")
                            results.append({"status": "skipped", "reason": "already exists"})
                            success = True
                            break
                        else:
                            errors.append(f"Payload {idx} failed (HTTP 400): {resp.text}")
                            break # Don't retry client errors
                    
                    elif resp.status_code >= 500:
                        print(f"[{idx}] Server error (HTTP {resp.status_code}). Retrying in 2s...")
                        time.sleep(2)
                    else:
                        errors.append(f"Payload {idx} failed (HTTP {resp.status_code}): {resp.text}")
                        break

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                print(f"[{idx}] Network error ({type(e).__name__}): {str(e)}. Retrying in 2s...")
                time.sleep(2)
            except Exception as e:
                print(f"[{idx}] Runtime error: {str(e)}")
                errors.append(f"Payload {idx} runtime error: {str(e)}")
                break

        if success and current_num < total_payloads:
            # ── Problem: Race Condition Fix ───────────────────────────
            # Add stabilization delay to allow Strapi to rebuild registry
            print(f"[ExecutionAgent] Waiting for Strapi stabilization (2s)")
            time.sleep(2)

    # ── Final State Update ─────────────────────────────────────────────
    state["execution_result"] = results if total_payloads > 1 else (results[0] if results else None)
    
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
