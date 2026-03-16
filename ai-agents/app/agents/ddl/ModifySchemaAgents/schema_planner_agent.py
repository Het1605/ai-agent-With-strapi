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
    previous_plan = state.get("modify_schema_plan", {})
    
    if not modify_operations and not previous_plan:
        print("[ModifySchemaPlannerAgent] No operations detected and no previous plan exists.")
        state["modify_schema_plan"] = {"operations": []}
        return state

    system_prompt = """
    You are a Senior Database Architect responsible for planning schema modifications.
    Your task is to convert detected schema modification intents into a structured modification plan.

    You will receive:
    1. Newly detected operations (from intent extraction)
    2. The previous schema modification plan (if this is an iterative refinement)
    3. The conversation history
    4. The latest user input

    --------------------------------------------------
    YOUR OBJECTIVE
    --------------------------------------------------
    Produce a structured JSON plan describing EXACTLY what schema modifications must be performed.
    Merge newly requested operations with the previous plan taking user revisions into account.

    Supported operations:
    1. add_column
    2. delete_column
    3. update_column
    4. update_collection

    --------------------------------------------------
    PLAN STRUCTURE (STRICT JSON)
    --------------------------------------------------
    Return ONLY valid JSON. Your response must be an object with an "operations" array.
    
    Do NOT include column types.
    Do NOT include constraints.
    Do NOT include relations.
    (The Designer Agent handles those details).
    
    Format:
    {
      "operations": [
        {
          "intent": "add_column",
          "table": "employees",
          "columns_to_add": [
            {"name": "salary"},
            {"name": "bonus"}
          ]
        },
        {
          "intent": "delete_column",
          "table": "employees",
          "columns_to_delete": [
            {"name": "leave_balance"}
          ]
        }
      ]
    }

    --------------------------------------------------
    ITERATIVE REFINEMENT RULES
    --------------------------------------------------
    If the user says something like "also add bonus column", you must RETAIN the previous operations in the plan and ADD the new one.
    If the user says "don't remove leave_balance", you must REMOVE that deletion from the plan.
    Support multiple tables and multiple operations gracefully.
    """

    context_message = f"""
    LATEST USER INPUT:
    "{user_input}"
    
    DETECTED NEW OPERATIONS (from Intent Agent):
    {json.dumps(modify_operations, indent=2)}
    
    PREVIOUS PLAN (Iterative State):
    {json.dumps(previous_plan, indent=2)}
    
    CONVERSATION HISTORY:
    {json.dumps(history, indent=2)}
    """

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_message)
        ])
        
        content = response.content.replace("```json", "").replace("```", "").strip()
        plan = json.loads(content)
        
        # Ensure correct base structure
        if "operations" not in plan:
            plan = {"operations": []}
            
        state["modify_schema_plan"] = plan
        print(f"[ModifySchemaPlannerAgent] Generated plan with {len(plan.get('operations', []))} operation groups.")
        
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
