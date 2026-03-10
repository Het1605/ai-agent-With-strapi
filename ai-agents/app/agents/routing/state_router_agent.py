from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState
import json

# Valid routes the agent is allowed to choose from.
# Extend this list when new DDL/DML leaf agents are added.
VALID_ROUTES = {"create_table", "modify_schema", "intent_router", "validation"}

async def state_router_agent(state: AgentState) -> AgentState:
    """
    StateRouterAgent — LLM-based routing decision after MemoryManager.

    Analyzes the current workflow state and decides which node should execute
    next, writing the result into state["route_decision"].

    Routing priorities (highest → lowest):
    1. interaction_phase == True  AND  active_agent is set
       → Resume the active agent directly (skip the full pipeline).
    2. current_task_index < len(task_queue)
       → Continue executing a planned task queue via intent_router.
    3. Otherwise
       → Fresh request; start the normal validation pipeline.
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
        "   This resumes a paused multi-turn conversation (e.g. asking the user for a missing table name).\n"
        "2. If current_task_index is less than task_queue_length AND interaction_phase is false → route to 'intent_router'.\n"
        "   This continues executing a planned task queue.\n"
        "3. Otherwise → route to 'validation'.\n"
        "   This starts the normal new-request pipeline.\n\n"
        f"ALLOWED ROUTES: {sorted(VALID_ROUTES)}\n\n"
        "Respond ONLY with a valid JSON object in this exact format:\n"
        "{\"route\": \"<chosen_route>\"}\n"
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

    # Parse LLM response
    chosen_route = "validation"  # safe default
    try:
        clean = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        raw_route = result.get("route", "validation").strip()

        if raw_route in VALID_ROUTES:
            print("Entering in If block")
            chosen_route = raw_route
            print(chosen_route)
        else:
            # If the LLM returned something outside the allowed set, fall back
            # but try to honour interaction_phase deterministically as a safety net.
            print(f"[StateRouterAgent] LLM returned unknown route '{raw_route}' — applying fallback logic.")
            if interaction_phase and active_agent in VALID_ROUTES:
                chosen_route = active_agent
            elif current_index < len(task_queue):
                chosen_route = "intent_router"
            else:
                chosen_route = "validation"

    except Exception as e:
        print(f"[StateRouterAgent] JSON parse error: {e} — defaulting to 'validation'.")
        # Apply deterministic fallback so parsing errors never stall the pipeline
        if interaction_phase and active_agent in VALID_ROUTES:
            chosen_route = active_agent
        elif current_index < len(task_queue):
            chosen_route = "intent_router"

    state["route_decision"] = chosen_route
    print(f"[StateRouterAgent] Decision → '{chosen_route}'")
    print(f"  interaction_phase  : {interaction_phase}")
    print(f"  active_agent       : {active_agent!r}")
    print(f"  task_queue_length  : {len(task_queue)}")
    print(f"  current_task_index : {current_index}")

    return state
