from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

# Valid routes StateRouterAgent may produce.
# NOTE: intent_router is intentionally EXCLUDED — it must only run after
# classification has resolved the scope.
VALID_ROUTES = {
    "create_table", "modify_schema",
    "add_column", "update_collection", "update_field", "delete_field",
    "validation"
}

async def state_router_agent(state: AgentState) -> AgentState:
    """
    StateRouterAgent — LLM-based routing decision after MemoryManager.

    Routing priorities (highest → lowest):
    1. interaction_phase == True  AND  active_agent is set
       → Resume the active DDL sub-agent directly (preserves mid-turn state).
    2. Otherwise
       → Route to 'validation' so the full pipeline runs:
         validation → classifier → intent_router → ddl/dml_router
    """
    print("\n----- ENTERING StateRouterAgent -----")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    interaction_phase = state.get("interaction_phase", False)
    active_agent      = state.get("active_agent") or ""
    task_queue        = state.get("task_queue") or []
    current_index     = state.get("current_task_index", 0)

    system_prompt = (
        "You are the StateRouterAgent of a multi-agent database management system.\n\n"
        "Your ONLY job is to decide which agent node should execute next.\n\n"
        "ROUTING RULES (apply in priority order):\n"
        "1. If interaction_phase is true AND active_agent is a non-empty string → route to that active_agent.\n"
        "   This resumes a paused multi-turn conversation (e.g. waiting for a missing table name).\n"
        "2. Otherwise → route to 'validation'.\n"
        "   This starts/continues the normal request pipeline:\n"
        "   validation → classifier → intent_router → ddl/dml_router.\n\n"
        "IMPORTANT: NEVER route to 'intent_router' directly.\n"
        "IntentRouterAgent must only run AFTER ScopeClassifierAgent.\n\n"
        f"ALLOWED ROUTES: {sorted(VALID_ROUTES)}\n\n"
        "Respond ONLY with valid JSON: {\"route\": \"<chosen_route>\"}\n"
        "Do NOT add explanation. Output ONLY the JSON."
    )

    human_prompt = (
        f"interaction_phase  : {interaction_phase}\n"
        f"active_agent       : {active_agent!r}\n"
        f"task_queue_length  : {len(task_queue)}\n"
        f"current_task_index : {current_index}\n"
        "\nWhich route should the workflow take?"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    # ── Parse LLM response ─────────────────────────────────────────────
    chosen_route = "validation"  # safe default
    try:
        clean     = response.content.replace("```json", "").replace("```", "").strip()
        raw_route = json.loads(clean).get("route", "validation").strip()

        if raw_route in VALID_ROUTES:
            chosen_route = raw_route
        else:
            print(f"[StateRouterAgent] LLM returned unknown route '{raw_route}' — applying deterministic fallback.")
            # Deterministic fallback: honour interaction_phase if active_agent is valid
            if interaction_phase and active_agent in VALID_ROUTES:
                chosen_route = active_agent
            # else: stay on 'validation'

    except Exception as e:
        print(f"[StateRouterAgent] JSON parse error: {e} — applying deterministic fallback.")
        if interaction_phase and active_agent in VALID_ROUTES:
            chosen_route = active_agent
        # else: stay on 'validation'

    state["route_decision"] = chosen_route
    print(f"[StateRouterAgent] Decision → '{chosen_route}'")
    print(f"  interaction_phase  : {interaction_phase}")
    print(f"  active_agent       : {active_agent!r}")
    print(f"  task_queue_length  : {len(task_queue)}")
    print(f"  current_task_index : {current_index}")

    return state
