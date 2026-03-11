from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

# Deterministic prefix → category map
_PREFIX_MAP = {
    "DDL_": "DDL",
    "DML_": "DML",
}


async def intent_router_agent(state: AgentState) -> AgentState:
    """
    IntentRouterAgent: categorises the user request into DDL or DML.
    
    Now supports direct inference from user_input (to bypass TaskPlannerAgent).

    Priority order:
      1. interaction_phase bypass        → DDL (resume active agent)
      2. Deterministic prefix matching   → DDL / DML (from current task if exists)
      3. LLM classification              → DDL / DML (from user_input or task_type)
    """
    print("\n----- ENTERING IntentRouterAgent -----")

    # ── 1. Interaction phase bypass ───────────────────────────────────
    if state.get("interaction_phase") == True:
        print("IntentRouterAgent: interaction_phase=True → bypassing, setting DDL.")
        state["intent_category"] = "DDL"
        return state

    user_input    = state.get("user_input", "")
    task_queue    = state.get("task_queue", [])
    current_index = state.get("current_task_index", 0)
    
    task_type = "UNKNOWN"
    if current_index < len(task_queue):
        current_task = task_queue[current_index]
        task_type = (current_task.get("task_type") or "UNKNOWN").strip().upper()
        print(f"Detected Task Type from queue: {task_type}")

    # ── 2. Deterministic prefix matching (no LLM) ─────────────────────
    if task_type != "UNKNOWN":
        for prefix, category in _PREFIX_MAP.items():
            if task_type.startswith(prefix):
                state["intent_category"] = category
                state["analysis"] = (state.get("analysis") or "") + \
                    f"\nIntent Router: {task_type} → {category} (deterministic)."
                print(f"Detected Category: {category} [deterministic]")
                return state

    # ── 3. LLM inference for classification ───────────────────────────
    # We now interpret either the task_type OR the raw user_input
    print(f"IntentRouterAgent: Calling LLM for intent classification.")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = (
        "You are an intent classifier for a database system.\n\n"
        "Your goal is to classify the user's request into one of these categories:\n"
        "  DDL — schema operations (create table, add field, modify collection, design system)\n"
        "  DML — data operations (insert, update, delete, select records)\n\n"
        "Rules:\n"
        "1. If the user wants to CREATE or CHANGE the structure of the database (tables, columns), respond DDL.\n"
        "2. If the user wants to MANIPULATE or VIEW experimental data (rows, records, entries), respond DML.\n"
        "3. Respond ONLY with: DDL or DML."
    )

    human_msg = f"Request: {task_type if task_type != 'UNKNOWN' else user_input}"
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    category = response.content.strip().upper()
    if category not in {"DDL", "DML"}:
        print(f"IntentRouterAgent: LLM returned unexpected value '{category}' — defaulting to DDL.")
        category = "DDL"

    state["intent_category"] = category
    state["analysis"] = (state.get("analysis") or "") + \
        f"\nIntent Router: {category} (LLM inference)."

    print(f"Detected Category: {category} [LLM inference]")
    return state
