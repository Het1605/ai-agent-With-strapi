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
    Upgraded to be context-aware (user_input + history).
    """
    print("\n----- ENTERING IntentRouterAgent (Context Aware) -----")

    # ── 1. Interaction phase bypass ───────────────────────────────────
    if state.get("interaction_phase") == True:
        print("IntentRouterAgent: interaction_phase=True → bypassing, setting DDL.")
        state["intent_category"] = "DDL"
        return state

    user_input    = state.get("user_input", "")
    history       = state.get("conversation_history", [])[-5:]
    task_queue    = state.get("task_queue", [])
    current_index = state.get("current_task_index", 0)
    
    print(f"[IntentRouter] User Input: {user_input}")

    task_type = "UNKNOWN"
    if current_index < len(task_queue):
        current_task = task_queue[current_index]
        task_type = (current_task.get("task_type") or "UNKNOWN").strip().upper()

    # ── 2. Deterministic prefix matching (no LLM) ─────────────────────
    if task_type != "UNKNOWN":
        for prefix, category in _PREFIX_MAP.items():
            if task_type.startswith(prefix):
                state["intent_category"] = category
                state["analysis"] = (state.get("analysis") or "") + \
                    f"\nIntent Router: {task_type} → {category} (deterministic)."
                print(f"[IntentRouter] Decision: {category} [deterministic]")
                return state

    # ── 3. LLM inference for classification ───────────────────────────
    print(f"[IntentRouter] Calling LLM for intent classification.")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = (
        "You are an intent classifier for a database AI system.\n\n"
        "Your job is to classify the user's request into one of two categories:\n\n"
        "DDL (Data Definition Language)\n"
        "Operations that CREATE or MODIFY database structure.\n"
        "Examples: create table, design database, add column, rename collection, delete field.\n\n"
        "DML (Data Manipulation Language)\n"
        "Operations that manipulate or retrieve stored records.\n"
        "Examples: insert record, add data, update record, delete entry, show entries, fetch students.\n\n"
        "Routing Rules:\n"
        "1. If the request modifies database structure (collections, fields) → DDL\n"
        "2. If the request works with actual data records/rows → DML\n"
        "3. System design requests (create full database/design system) → DDL\n"
        "4. Requests referencing 'entries', 'rows', or 'records' specifically → DML\n"
        "5. Follow-up requests must consider conversation history.\n\n"
        "Respond ONLY with exactly one word: DDL or DML."
    )

    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    human_msg = (
        f"Conversation History:\n{history_str}\n\n"
        f"Current User Request: {user_input}"
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg)
    ])

    category = response.content.strip().upper()
    if category not in {"DDL", "DML"}:
        print(f"[IntentRouter] LLM returned unexpected value '{category}' — defaulting to DDL.")
        category = "DDL"

    state["intent_category"] = category
    state["analysis"] = (state.get("analysis") or "") + \
        f"\nIntent Router: {category} (context-aware)."

    print(f"[IntentRouter] Decision: {category}")
    return state
