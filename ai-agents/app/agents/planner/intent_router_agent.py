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
    IntentRouterAgent: categorises the current task into DDL or DML.

    Priority order:
      1. interaction_phase bypass        → DDL (resume active agent)
      2. Empty task queue                → FORMATTER
      3. Deterministic prefix matching   → DDL / DML  (no LLM call)
      4. LLM fallback                    → DDL / DML  (unknown task_type only)
    """
    print("\n----- ENTERING IntentRouterAgent -----")

    # ── 1. Interaction phase bypass ───────────────────────────────────
    if state.get("interaction_phase") == True:
        print("IntentRouterAgent: interaction_phase=True → bypassing, setting DDL.")
        state["intent_category"] = "DDL"
        return state

    task_queue    = state.get("task_queue", [])
    current_index = state.get("current_task_index", 0)
    print(f"Current Task Index: {current_index}")

    # ── 2. Empty queue guard ──────────────────────────────────────────
    if current_index >= len(task_queue):
        print("IntentRouterAgent: Task queue empty → routing to formatter.")
        state["intent_category"] = "FORMATTER"
        return state

    current_task = task_queue[current_index]
    task_type    = (current_task.get("task_type") or "UNKNOWN").strip().upper()
    print(f"Task Type: {task_type}")

    # ── 3. Deterministic prefix matching (no LLM) ─────────────────────
    for prefix, category in _PREFIX_MAP.items():
        if task_type.startswith(prefix):
            state["intent_category"] = category
            state["analysis"] = (state.get("analysis") or "") + \
                f"\nIntent Router: {task_type} → {category} (deterministic)."
            print(f"Detected Category: {category}  [deterministic]")
            return state

    # ── 4. LLM fallback for unknown task types ────────────────────────
    print(f"IntentRouterAgent: Unknown task_type '{task_type}', calling LLM fallback.")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = (
        "You are an intent classifier for a database system.\n\n"
        "You will receive a TASK TYPE.\n"
        "You must classify it into:\n"
        "  DDL — schema operations\n"
        "  DML — data operations\n\n"
        "Valid DDL tasks: DDL_CREATE_TABLE, DDL_MODIFY_SCHEMA\n"
        "Valid DML tasks: DML_INSERT, DML_UPDATE, DML_DELETE, DML_SELECT\n\n"
        "Rules:\n"
        "1. Use ONLY the provided TASK TYPE.\n"
        "2. Do NOT interpret natural language.\n"
        "3. If the task starts with 'DDL_' respond with DDL.\n"
        "4. If the task starts with 'DML_' respond with DML.\n\n"
        "Respond ONLY with: DDL\nor: DML"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Classify this task type: {task_type}")
    ])

    category = response.content.strip().upper()
    if category not in {"DDL", "DML"}:
        print(f"IntentRouterAgent: LLM returned unexpected value '{category}' — defaulting to DDL.")
        category = "DDL"

    state["intent_category"] = category
    state["analysis"] = (state.get("analysis") or "") + \
        f"\nIntent Router: {task_type} → {category} (LLM fallback)."

    print(f"Detected Category: {category}  [LLM fallback]")
    return state
