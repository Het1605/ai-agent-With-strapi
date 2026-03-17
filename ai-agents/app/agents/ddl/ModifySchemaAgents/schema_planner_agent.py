import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.state import AgentState

async def schema_planner_agent(state: AgentState) -> AgentState:
    """
    ModifySchemaPlannerAgent: Converts detected modification intents and history
    into a structured schema modification plan for the designer agent.
    """
    print("\n----- ENTERING ModifySchemaPlannerAgent -----")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    user_input = state.get("user_input", "")
    history = state.get("conversation_history", [])
    modify_operations = state.get("modify_operations", [])
    
    memory = state.get("modify_schema_memory", {
        "planner_history": [],
        "designer_history": [],
        "latest_plan": {"operations": []},
        "latest_design": {"operations": []},
        "iteration_count": 0
    })
    previous_plan = memory.get("latest_plan", {})
    existing_collections = state.get("existing_collections", [])
    existing_schema = state.get("existing_schema", {})

    print("existing_collections:",existing_collections)

    print("modify_operations:",modify_operations)

    print("previous_plan:",previous_plan)
    
    if not modify_operations and not previous_plan:
        print("[ModifySchemaPlannerAgent] No operations detected and no previous plan exists.")
        state["modify_schema_plan"] = {"operations": []}
        return state

    system_prompt = """
    You are a Senior Database Architect acting as the ModifySchemaPlannerAgent.
    Your objective is to design production-grade database schema modifications by interpreting user intent intelligently and continuously adapting to state.

    You will receive:
    1. Newly detected operations (from intent extraction)
    2. The entire modify_schema_memory (containing history of previous plans)
    3. The conversation history
    4. The latest user input
    5. The existing_schema for all collections

    --------------------------------------------------
    YOUR CORE RESPONSIBILITIES
    --------------------------------------------------
    1. INTELLIGENT INTENT REASONING
       - Analyze what the user is truly asking:
         * Enhancement (add useful, domain-relevant fields)
         * Cleanup (remove unnecessary/weak fields)
         * Correction (update existing fields)
         * Restructure (major overhaul or shifting design)
       - DO NOT depend on keywords. Think like a real architect.
       - NEVER return an empty plan. If the user request is vague (e.g., "add work fields"), infer the domain from the table name (e.g., employee -> HR domain) and propose a robust, realistic set of columns (e.g., designation, department, salary).

    2. STATEFUL MEMORY & CONTEXT AWARENESS
       - You are working in a continuous session. Read `modify_schema_memory` and `conversation_history`.
       - Understand the delta between the previous iteration and the new user request.
       - Decide internally whether the user wants to ADD to the previous design, MODIFY it, or DISCARD it (e.g., if they say "start fresh", you naturally drop the previous plan operations; if they say "also add", you logically append to the previous plan operations).
       - DO NOT output any explicit keyword like "action: MERGE" or "action: RESET". Simply output the FULL, FINAL set of operations that represents the current desired state.

    3. STRICT SCHEMA AWARENESS
       - Validate all your decisions against `existing_schema`.
       - DO NOT invent duplicate columns.
       - DO NOT assume a field exists before deleting or updating it.
       - Do not add relation columns to tables that do not exist.

    --------------------------------------------------
    COLLECTION-LEVEL MODIFICATIONS (update_collection)
    --------------------------------------------------
    - Permitted ONLY for updating `displayName` or deleting a table entirely (high risk).
    - To rename: `{"displayName": "New Name"}`.
    - To delete: `{"delete": true}` ONLY if explicitly requested (e.g., "drop table", "remove collection").

    --------------------------------------------------
    PLAN STRUCTURE (STRICT JSON)
    --------------------------------------------------
    Return ONLY valid JSON. Your response must be an object containing ONLY an "operations" array.
    No explanations, no wrapper blocks.

    Format:
    {
      "operations": [
        {
          "intent": "add_column",
          "table": "employees",
          "columns_to_add": [{"name": "salary"}]
        }
      ]
    }
    """

    context_message = f"""
    LATEST USER INPUT:
    "{user_input}"
    
    DETECTED NEW OPERATIONS (from Intent Agent):
    {json.dumps(modify_operations, indent=2)}
    
    SYSTEM MEMORY (History of Previous Plans):
    {json.dumps(memory, indent=2)}

    EXISTING COLLECTIONS SCHEMA:
    {json.dumps(existing_schema, indent=2) if existing_schema else "[]"}
    
    CONVERSATION HISTORY:
    {json.dumps(history, indent=2)}
    """

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_message)
        ])
        
        content = response.content.replace("```json", "").replace("```", "").strip()
        parsed_json = json.loads(content)

        print("Planner Content:",content)
        
        plan_ops = parsed_json.get("operations", [])
        plan = {"operations": plan_ops}
            
        memory["planner_history"].append(plan)
        memory["latest_plan"] = plan
        memory["iteration_count"] += 1
        
        state["modify_schema_memory"] = memory
        state["modify_schema_plan"] = plan
        
        print(f"[ModifySchemaPlannerAgent] Generated comprehensive plan with {len(plan_ops)} operation groups.")
        
    except json.JSONDecodeError as e:
        print(f"[ModifySchemaPlannerAgent] JSON parse error: {e}")
        # If it fails, fallback to passing through previous or empty
        if not state.get("modify_schema_plan"):
            state["modify_schema_plan"] = {"operations": []}
    except Exception as e:
        print(f"[ModifySchemaPlannerAgent] Unexpected error: {e}")
        if not state.get("modify_schema_plan"):
            state["modify_schema_plan"] = {"operations": []}

    return state
